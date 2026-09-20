# gpu_spacetime.jl — generic-spacetime fused GPU imaging: unpolarized Stokes I, full
# IQUV, and a no-fluid shadow/capture tracer. The device metric layer is the CPU
# geodesics layer compiled in-kernel: isbits spacetimes (`SuperposedKerrSchild`) pass as
# kernel parameters and the kernels call the module originals `metric` / `inv` /
# `geodesic_rhs` / `rk4_step_fixed`; `ADMGridSpacetime` is mirrored by `DevADMParams`
# (same field names, device data array) through the same `_adm_gcov` body. The fluid is
# sampled via the O(nmb) scan locate (`_gamr_locate_scan` + `_gamr_lerp`) for meshes
# beyond the dense leaf-table cap. Float64 only. CUDA optional: NOT included from
# BlackLightPlus.jl; include AFTER gpu_image/gpu_pol/gpu_amr (auto-included if absent).

using StaticArrays
using ForwardDiff: Dual, partials, value
using KernelAbstractions
using CUDA
using Adapt
using LinearAlgebra: dot, norm
using BlackLightPlus: SuperposedKerrSchild, PlaneParallelCamera, camera_rays
using BlackLightPlus.Geodesics: metric, inverse_metric, geodesic_rhs, rk4_step_fixed,
      ks_radius_hole, horizon_radius_hole, HORIZON_EPS_SAMPLED, ADMGridSpacetime,
      StationarySpacetime, _adm_gcov
using BlackLightPlus.Geodesics: BinaryOrbit, SuperposedBoostedKerrSchildCartesian
using BlackLightPlus.Simulation: AthenaKSnapshot

isdefined(@__MODULE__, :_gimg_coeffs) || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_image.jl"))
isdefined(@__MODULE__, :_gpol_march)  || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_pol.jl"))
isdefined(@__MODULE__, :_gamr_lerp)   || Base.include(@__MODULE__, joinpath(@__DIR__, "gpu_amr.jl"))

# ---------------------------------------------------------------- device ADM params
"""
    DevADMParams(st::ADMGridSpacetime, data) <: StationarySpacetime

GPU twin of `ADMGridSpacetime`: the same grid-geometry fields (same names, so the
duck-typed `sample_adm`/`on_grid`/`_adm_gcov` bodies run unchanged) with `data` swapped
for a device array. Subtyping `StationarySpacetime` makes the CPU engine methods apply
verbatim, so the device ADM derivative semantics are exactly the CPU's: 3-lane
hand-seeded duals through `inv(_adm_gcov(·))`, C¹ Catmull-Rom weights carrying the Dual.
"""
struct DevADMParams{A} <: StationarySpacetime{Float64}
    x0::Float64; y0::Float64; z0::Float64
    dx::Float64; dy::Float64; dz::Float64
    nx::Int; ny::Int; nz::Int
    data::A
    alpha_min::Float64
    offgrid_eta::Bool
end
DevADMParams(st::ADMGridSpacetime, data; alpha_min::Real = st.alpha_min) =
    DevADMParams(st.x0, st.y0, st.z0, st.dx, st.dy, st.dz, st.nx, st.ny, st.nz,
                 data, Float64(alpha_min), st.offgrid_eta)
Adapt.adapt_structure(to, p::DevADMParams) =
    DevADMParams(p.x0, p.y0, p.z0, p.dx, p.dy, p.dz, p.nx, p.ny, p.nz,
                 Adapt.adapt(to, p.data), p.alpha_min, p.offgrid_eta)

# The SAME ADM assembly body as the CPU spacetime (single source, numerical_spacetime.jl).
BlackLightPlus.Geodesics.metric(sp::DevADMParams, X::SVector{4}) = _adm_gcov(sp, X)
BlackLightPlus.Geodesics.inverse_metric(sp::DevADMParams, X::SVector{4}) = inv(_adm_gcov(sp, X))

"""
    upload_adm(st::ADMGridSpacetime, devices; alpha_min=st.alpha_min)
        -> Dict{Int,DevADMParams}

One `DevADMParams` per CUDA device (the (10,nx,ny,nz) field array uploaded per device).
`alpha_min` overrides the capture lapse (e.g. the z4c shadow figure uses 0.65 on a
grid built with 0.15).
"""
function upload_adm(st::ADMGridSpacetime, devices; alpha_min::Real = st.alpha_min)
    d = Dict{Int,Any}()
    for dev in devices
        CUDA.device!(dev)
        d[dev] = DevADMParams(st, CuArray(st.data); alpha_min = alpha_min)
    end
    return d
end

# ---------------------------------------------------------------- march control
# Per-spacetime fixed-control march control: (captured, step h, escape-test radius).
# SuperposedKerrSchild mirrors the CPU tracer (spacetimes.jl): step by the distance to
# the NEAREST horizon h = min_A(r_A − r_h_A)/C, capture (either hole) at horizon +
# HORIZON_EPS_SAMPLED, escape on the global coordinate radius. Non-finite r_A ⇒
# dmin = −Inf ⇒ capture, as the CPU loop.
@inline function _gsp_ctl(st::SuperposedKerrSchild{N,Float64}, y::SVector{8,Float64},
                          C::Float64) where {N}
    x = SVector(y[1], y[2], y[3], y[4])
    dmin = Inf
    @inbounds for A in 1:N
        dA = ks_radius_hole(st, x, A) - horizon_radius_hole(st, A)
        isfinite(dA) || (dmin = -Inf; break)
        dmin = min(dmin, dA)
    end
    R = sqrt(y[2] * y[2] + y[3] * y[3] + y[4] * y[4])
    return dmin < HORIZON_EPS_SAMPLED, dmin / C, R
end

# ADM grid mirrors the z4c shadow tracer: capture when the local lapse
# α = 1/√(−g^{tt}) drops below alpha_min (off-grid the metric is Minkowski ⇒ α = 1,
# never captures), coordinate-radius step h = max(r,1)/C, escape on the coordinate
# radius. A non-finite state (puncture blow-up) is a capture.
@inline function _gsp_ctl(sp::DevADMParams, y::SVector{8,Float64}, C::Float64)
    x = SVector(y[1], y[2], y[3], y[4])
    r = sqrt(y[2] * y[2] + y[3] * y[3] + y[4] * y[4])
    gtt = inverse_metric(sp, x)[1, 1]
    alpha = gtt < 0.0 ? 1.0 / sqrt(-gtt) : 1.0
    return (alpha < sp.alpha_min) | !isfinite(r), max(r, 1.0) / C, r
end

# ---------------------------------------------------------------- generic coeffs
# One sample -> (j_inv, α_inv) for an arbitrary metric: the `_gimg_coeffs` physics with
# (gcov, gcon) passed in instead of the Kerr closed form. Deliberately a sibling of
# `_gimg_coeffs`, not a refactor — the Kerr body is pinned bit-exact against
# fmuladd-contraction drift (see its NOTE).
@inline function _gimg_coeffs_st(prim::SVector{8,Float64}, kcov::SVector{4,Float64},
                                 gcov::SMatrix{4,4}, gcon::SMatrix{4,4}, gp)
    z = zero(promote_type(typeof(gp.d_unit), typeof(gp.e_unit), typeof(gp.b_unit),
                          typeof(gp.rat_high), typeof(gp.rat_low)))
    rho = prim[1]; eint = prim[5]
    (rho <= 0.0 || eint <= 0.0) && return (z, z)
    pgas = (gp.gamma - 1.0) * eint
    ucon, ucov, bcon, b_sq = _gimg_pfv(prim[2], prim[3], prim[4],
                                       prim[6], prim[7], prim[8], gcov, gcon)
    b_sq <= 0.0 && return (z, z)
    rho_cgs = rho * gp.d_unit
    pgas_cgs = pgas * gp.e_unit
    n_cgs = rho_cgs / (gp.mu * _MP)
    n_e = n_cgs / (1.0 + 1.0 / gp.ne_ni)
    bb_cgs = sqrt(b_sq) * gp.b_unit
    sigma = b_sq / rho
    beta_inv = b_sq / (2.0 * pgas)
    theta_e = _gimg_theta_e(rho_cgs, pgas_cgs, beta_inv, gp)
    (gp.sigma_cut >= 0.0 && sigma > gp.sigma_cut) && return (z, z)
    (gp.theta_e_min >= 0.0 && theta_e < gp.theta_e_min) && return (z, z)
    minus_ku = -dot(kcov, ucon)                 # = ν_local / ν_obs (p_t = -1)
    kb = dot(kcov, bcon)
    cos2 = min(kb * kb / (minus_ku * minus_ku * b_sq), 1.0)
    sin_tb = sqrt(1.0 - cos2)
    nu_local = minus_ku * gp.nu_obs
    th = _gimg_thermal(gp.thermal_frac, n_e, theta_e, bb_cgs, nu_local, sin_tb)
    j = th[1]; a = th[2]
    if hasfield(typeof(gp), :power_frac)
        jp, ap = _gimg_powerlaw(gp, n_e, bb_cgs, nu_local, sin_tb)
        j += jp; a += ap
    end
    if hasfield(typeof(gp), :kappa_frac)
        jk, ak = _gimg_kappa(gp, n_e, bb_cgs, nu_local, sin_tb)
        j += jk; a += ak
    end
    return (j, a)
end

# ---------------------------------------------------------------- generic unpol march
# The generic-spacetime fused unpolarized march for one ray, camera->source: the
# `_gimg_march` reverse-accumulation scheme (gpu_image.jl header) with the geodesic
# step, step control and termination supplied by the spacetime (`_gsp_ctl` +
# `rk4_step_fixed`/`geodesic_rhs`, the CPU engine in-kernel). Float64-only (the fp32
# critical-curve rescue is a Kerr-specific construct).
@inline function _gsp_march(y0::SVector{8,Float64}, sample::F, gp, sp,
                            C::Float64, escape::Float64, R_em::Float64, nmax) where {F}
    y = y0
    Iinv = 0.0
    trans = 1.0
    r_prev = Inf
    for _ in 1:nmax
        cap, h, re = _gsp_ctl(sp, y, C)
        cap && break
        ((re > escape || re > R_em) && re > r_prev) && break   # outbound-only, as CPU
        r_prev = re
        prim = @inline sample(y)
        if prim[1] > 0.0
            kcov = SVector(y[5], y[6], y[7], y[8])
            gcov = metric(sp, SVector(y[1], y[2], y[3], y[4]))
            gcon = inv(gcov)          # == inverse_metric(sp, x) (both generic spacetimes)
            j_inv, a_inv = _gimg_coeffs_st(prim, kcov, gcov, gcon, gp)
            ds = h * gp.L_unit / gp.nu_obs
            A, B = _gimg_ab(j_inv, a_inv, ds)
            Iinv += trans * B
            trans *= A
        end
        y = rk4_step_fixed(sp, y, h)
    end
    return Iinv * gp.nu_obs^3
end

# ---------------------------------------------------------------- generic pol frame
# Renormalizing tetrad for arbitrary metrics: the pinned Kerr device body `_gpol_tetrad`
# (gpu_pol.jl) + the explicit e_(1) renormalization of the CPU `fluid_tetrad(...;
# renorm_e1 = true)` — e_(1) is raised from the metric cross product with norm
# 1/|det g|, unit only on single CKS (|det g| = 1), so every spacetime marched here
# needs the renorm, keyed as the CPU marches on `!(st isa KerrSchildCartesian)`.
@inline function _gsp_tetrad(ucon::SVector{4}, ucov::SVector{4}, kcon::SVector{4},
                             kcov::SVector{4}, up::SVector{4},
                             gcov::SMatrix{4,4}, gcon::SMatrix{4,4})
    tet = _gpol_tetrad(ucon, ucov, kcon, kcov, up, gcov, gcon)
    e1 = SVector(tet[2, 1], tet[2, 2], tet[2, 3], tet[2, 4])
    e1_norm = sqrt(dot(e1, gcov * e1))
    if e1_norm > 0.0
        e1 = e1 ./ e1_norm
    end
    return @inbounds SMatrix{4,4}(tet[1, 1], e1[1], tet[3, 1], tet[4, 1],
                                  tet[1, 2], e1[2], tet[3, 2], tet[4, 2],
                                  tet[1, 3], e1[3], tet[3, 3], tet[4, 3],
                                  tet[1, 4], e1[4], tet[3, 4], tet[4, 4])
end

# ---------------------------------------------------------------- generic extended RHS
# Geodesic RHS + parallel transport of TWO frame legs from ONE shared 3-lane AD pass
# through the spacetime's covariant metric — the `_gpol_ext_rhs2` structure (gpu_pol.jl)
# with `metric(sp, ·)` in place of the Kerr closed form. The value lane of `inv` on a
# Dual matrix performs the identical scalar ops as `inv` on the value matrix, and the
# CPU `inverse_metric` of both generic spacetimes IS `inv(metric)`, so the geodesic lane
# below is bit-identical to the CPU engine stepping used by `_gsp_march`. The partials
# of gcovd give ∂_c g_{ab} (c = 2,3,4; ∂_t g = 0, stationary slice) for the transport
# deV^μ/dλ = -Γ^μ_{αβ} k^α V^β. Float64-only.
@inline function _gsp_ext_rhs2(sp, y::SVector{8,Float64}, eU::SVector{4,Float64},
                               eV::SVector{4,Float64})
    x = SVector(y[1], y[2], y[3], y[4]); p = SVector(y[5], y[6], y[7], y[8])
    z = 0.0; o = 1.0
    xd = SVector(Dual(x[1], (z, z, z)), Dual(x[2], (o, z, z)),
                 Dual(x[3], (z, o, z)), Dual(x[4], (z, z, o)))
    gcovd = metric(sp, xd)
    gcond = inv(gcovd)
    gcon = SMatrix{4,4,Float64}(ntuple(k -> value(gcond[k]), Val(16)))
    kcon = gcon * p
    dp1 = z; dp2 = z; dp3 = z
    @inbounds for μ in 1:4, ν in 1:4
        pp = p[μ] * p[ν]; prt = partials(gcond[μ, ν])
        dp1 -= pp * prt[1]; dp2 -= pp * prt[2]; dp3 -= pp * prt[3]
    end
    dy = SVector(kcon[1], kcon[2], kcon[3], kcon[4], z, dp1 / 2, dp2 / 2, dp3 / 2)
    dg2 = SMatrix{4,4,Float64}(ntuple(k -> partials(gcovd[k])[1], Val(16)))
    dg3 = SMatrix{4,4,Float64}(ntuple(k -> partials(gcovd[k])[2], Val(16)))
    dg4 = SMatrix{4,4,Float64}(ntuple(k -> partials(gcovd[k])[3], Val(16)))
    Dk = kcon[2] .* dg2 .+ kcon[3] .* dg3 .+ kcon[4] .* dg4
    @inline function _transport(v)
        Dv = v[2] .* dg2 .+ v[3] .* dg3 .+ v[4] .* dg4
        t1 = Dk * v; t2 = Dv * kcon
        t3 = SVector(z, dot(kcon, dg2 * v), dot(kcon, dg3 * v), dot(kcon, dg4 * v))
        return -(0.5 .* (gcon * (t1 .+ t2 .- t3)))
    end
    return dy, _transport(eU), _transport(eV)
end

# ---------------------------------------------------------------- generic pol march
# The generic-spacetime fused POLARIZED march for one ray, camera->source: the Kerr
# reverse-Mueller scheme `_gpol_march` (gpu_pol.jl) with (i) march control/termination
# from the spacetime (`_gsp_ctl`), (ii) the RT-lane metric from `metric(sp, ·)` + `inv`,
# (iii) geodesic + two-leg frame transport stepped by the fused low-storage RK4 over
# `_gsp_ext_rhs2` (same combine ops as `rk4_step_fixed`, so the trajectory is
# bit-compatible with the unpol march / CPU tracer), and (iv) the renormalizing
# `_gsp_tetrad` in both the fluid tetrad and the readout frame. The metric-independent
# pieces (`_gpol_coeffs`/`_gpol_segMB`/`_gpol_R`/`_gpol_admissible`) are the gpu_pol.jl
# originals — no duplicated formula bodies.
@inline function _gsp_pol_march(y0::SVector{8,Float64}, sample::F, gp, sp,
                                C::Float64, escape::Float64, R_em::Float64,
                                nmax) where {F}
    y = y0
    x0 = SVector(y0[1], y0[2], y0[3], y0[4]); kcov0 = SVector(y0[5], y0[6], y0[7], y0[8])
    gcov0 = metric(sp, x0); gcon0 = inv(gcov0); kcon0 = gcon0 * kcov0
    e0c = _gpol_normal_observer(gcon0)
    e0cov0 = gcov0 * e0c
    cu = SVector(gp.cu1, gp.cu2, gp.cu3, gp.cu4)
    upc0 = e0c[1]*cu[1] - (e0cov0[2]*cu[2] + e0cov0[3]*cu[3] + e0cov0[4]*cu[4]) / e0cov0[1]
    upcam = SVector(upc0, cu[2]+e0c[2]*cu[1], cu[3]+e0c[3]*cu[1], cu[4]+e0c[4]*cu[1])
    Ecam = _gsp_tetrad(e0c, e0cov0, kcon0, kcov0, upcam, gcov0, gcon0)
    eU  = SVector(Ecam[3, 1], Ecam[3, 2], Ecam[3, 3], Ecam[3, 4])   # camera e_(2) (up)
    eUu = e0c                    # camera 4-velocity, parallel-transported alongside eU
    S = _gpz4; G = _gpI4
    r_prev = Inf
    for _ in 1:nmax
        cap, h, re = _gsp_ctl(sp, y, C)
        cap && break
        ((re > escape || re > R_em) && re > r_prev) && break   # outbound-only, as CPU
        r_prev = re
        prim = @inline sample(y)
        if prim[1] > 0.0
            x = SVector(y[1], y[2], y[3], y[4])
            gcov = metric(sp, x); gcon = inv(gcov)
            kcov = SVector(y[5], y[6], y[7], y[8]); kcon = gcon * kcov
            j, al, ro, tet_fl, valid = _gpol_coeffs(prim, x, kcov, gcov, gcon, gp,
                                                    _gsp_tetrad)
            if valid
                E = _gsp_tetrad(eUu, gcov * eUu, kcon, kcov, eU, gcov, gcon)
                ds = h * gp.L_unit / gp.nu_obs
                M, b = _gpol_segMB(j, al, ro, ds)
                R = _gpol_R(tet_fl, E, gcov)
                S = _gpol_admissible(S + G * (R * b))   # CPU stokes_couple clamp mirror
                G = G * (R * M * transpose(R))
            end
        end
        # low-storage streaming RK4 of {y, eU, eUu} (the _gpol_march scheme; same
        # ops/associativity as rk4_step_fixed's 4-stage combine)
        dy, de, du = _gsp_ext_rhs2(sp, y, eU, eUu)
        ay = dy; ae = de; au = du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, de, du = _gsp_ext_rhs2(sp, ys, es, us)
        ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
        ys = y + (h/2)*dy; es = eU + (h/2)*de; us = eUu + (h/2)*du
        dy, de, du = _gsp_ext_rhs2(sp, ys, es, us)
        ay = ay + 2*dy; ae = ae + 2*de; au = au + 2*du
        ys = y + h*dy; es = eU + h*de; us = eUu + h*du
        dy, de, du = _gsp_ext_rhs2(sp, ys, es, us)
        ay = ay + dy; ae = ae + de; au = au + du
        y   = y   + (h/6)*ay
        eU  = eU  + (h/6)*ae
        eUu = eUu + (h/6)*au
    end
    return S * gp.nu_obs^3
end

# ---------------------------------------------------------------- device: scan sample
# Scan-locate AMR sample: the exact `_find_meshblock_scan` port (`_gamr_locate_scan`,
# later block wins, 0 outside) + the shared in-block `_gamr_lerp` — the CPU
# `sample_primitives` route on meshes whose finest-level grid exceeds the leaf-table
# cap. O(nmb) per sample.
@inline function _gamr_sample_scan(blk, xlft, dxb, lo, hi, ni, nj, nk, nmb, x, y, z)
    mb = _gamr_locate_scan(lo, hi, nmb, x, y, z)
    mb == 0 && return zero(SVector{8,Float64})
    return @inline _gamr_lerp(blk, xlft, dxb, ni, nj, nk, mb, x, y, z)
end

# ---------------------------------------------------------------- device scan grid
"""
    DeviceAMRScanGrid

Per-device uploaded AMR fluid + per-block face bounds for the SCAN-locate sampler
(no leaf table — for meshes beyond the dense-grid cap where `upload_amr` errors).
Built by `upload_amr_scan`; consumed by the generic-spacetime kernels.
"""
struct DeviceAMRScanGrid
    dev_blk::Dict{Int,CuArray{Float32,5}}
    dev_xlft::Dict{Int,CuArray{Float64,2}}
    dev_dxb::Dict{Int,CuArray{Float64,2}}
    dev_lo::Dict{Int,CuArray{Float64,2}}
    dev_hi::Dict{Int,CuArray{Float64,2}}
    ni::Int; nj::Int; nk::Int; nmb::Int
end

"""
    upload_amr_scan(snap::AthenaKSnapshot, devices) -> DeviceAMRScanGrid

Permute the reader's primitives to the coalesced `(8,ni,nj,nk,nmb)` layout and upload
fluid + per-block geometry (face bounds for the scan locate) to each device. Works for
ANY mesh (no leaf-table requirement); reuse across renders.
"""
function upload_amr_scan(snap::AthenaKSnapshot, devices)
    blk_host  = permutedims(snap.data, (5, 1, 2, 3, 4))    # (8, ni, nj, nk, nmb)
    xlft_host = reduce(hcat, snap.x_left)                   # (3, nmb)
    dxb_host  = reduce(hcat, snap.dx)
    lo_host   = reduce(hcat, snap.ext_lo)
    hi_host   = reduce(hcat, snap.ext_hi)
    dblk = Dict{Int,CuArray{Float32,5}}(); dxl = Dict{Int,CuArray{Float64,2}}()
    ddx  = Dict{Int,CuArray{Float64,2}}(); dlo = Dict{Int,CuArray{Float64,2}}()
    dhi  = Dict{Int,CuArray{Float64,2}}()
    for dev in devices
        CUDA.device!(dev)
        dblk[dev] = CuArray(blk_host); dxl[dev] = CuArray(xlft_host)
        ddx[dev]  = CuArray(dxb_host); dlo[dev] = CuArray(lo_host)
        dhi[dev]  = CuArray(hi_host)
    end
    return DeviceAMRScanGrid(dblk, dxl, ddx, dlo, dhi, snap.ni, snap.nj, snap.nk, snap.nmb)
end

# ---------------------------------------------------------------- kernels
# Generic-spacetime fused unpol kernel over the scan-locate AMR fluid. One ray/thread;
# `sp` is the isbits spacetime parameter (the CPU struct itself, or DevADMParams).
@kernel function _gsp_amr_kernel!(img, @Const(y0s), @Const(blk), @Const(xlft), @Const(dxb),
                                  @Const(lo), @Const(hi), sp, gp, C, escape, R_em,
                                  nmax, ni, nj, nk, nmb, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        img[i] = _gsp_march(y0s[i],
                            y -> _gamr_sample_scan(blk, xlft, dxb, lo, hi, ni, nj, nk,
                                                   nmb, y[2], y[3], y[4]),
                            gp, sp, C, escape, R_em, nmax)
    end
end

# Generic-spacetime fused POLARIZED kernel over the scan-locate AMR fluid (the
# _gsp_amr_kernel! twin with _gsp_pol_march; out[i] is the camera-frame Stokes 4-vector).
@kernel function _gsp_pol_amr_kernel!(out, @Const(y0s), @Const(blk), @Const(xlft),
                                      @Const(dxb), @Const(lo), @Const(hi), sp, gp, C,
                                      escape, R_em, nmax, ni, nj, nk, nmb, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        out[i] = _gsp_pol_march(y0s[i],
                                y -> _gamr_sample_scan(blk, xlft, dxb, lo, hi, ni, nj,
                                                       nk, nmb, y[2], y[3], y[4]),
                                gp, sp, C, escape, R_em, nmax)
    end
end

# No-fluid shadow/capture tracer (z4c lensing masks; puncture-capture + null gates).
# Mirrors the CPU fixed-control loop: per step — capture test, PLAIN escape (the camera
# sits inside the escape sphere on these decks), fixed-control step, non-finite state
# ⇒ capture. Tracks max |g^{μν} p_μ p_ν| (the CPU `null_error`) along the ray.
# out_m[i] = (status: 1 capture / 2 escape / 0 nmax, steps, max|k·k|, r_final).
@kernel function _gsp_shadow_kernel!(out_y, out_m, @Const(y0s), sp, C, escape, nmax, nrays)
    i = @index(Global)
    @inbounds if i <= nrays
        y = y0s[i]
        status = 0.0
        maxnull = 0.0
        n = 0
        for _ in 1:nmax
            cap, h, re = _gsp_ctl(sp, y, C)
            x = SVector(y[1], y[2], y[3], y[4])
            p = SVector(y[5], y[6], y[7], y[8])
            gi = inverse_metric(sp, x)
            nl = abs(dot(p, gi * p))
            isfinite(nl) && (maxnull = max(maxnull, nl))
            if cap
                status = 1.0; break
            end
            if re > escape
                status = 2.0; break
            end
            y = rk4_step_fixed(sp, y, h)
            n += 1
            if !(isfinite(y[1]) && isfinite(y[2]) && isfinite(y[3]) && isfinite(y[4]) &&
                 isfinite(y[5]) && isfinite(y[6]) && isfinite(y[7]) && isfinite(y[8]))
                status = 1.0; break      # blew up near a puncture => capture (trace_bbh)
            end
        end
        out_y[i] = y
        out_m[i] = SVector(status, Float64(n), maxnull,
                           sqrt(y[2] * y[2] + y[3] * y[3] + y[4] * y[4]))
    end
end

# ---------------------------------------------------------------- host drivers
# The per-device spacetime parameter: isbits spacetimes are shared as-is; an
# ADMGridSpacetime is uploaded per device (or pass the `upload_adm` Dict directly).
_gsp_per_device(st, devices) = Dict{Int,Any}(dev => st for dev in devices)
_gsp_per_device(st::ADMGridSpacetime, devices) = upload_adm(st, devices)
_gsp_per_device(d::Dict, devices) = d

function _gsp_part!(img, rays, rng, sg::DeviceAMRScanGrid, dev, sp, gp, C, escape,
                    R_em, nmax, wg)
    backend = CUDABackend()
    kern = _gsp_amr_kernel!(backend)
    cnt = length(rng)
    y0d = CuArray(rays[rng])
    outd = CuArray{Float64}(undef, cnt)
    _kern_launch!(kern, wg, cnt, outd, y0d, sg.dev_blk[dev], sg.dev_xlft[dev],
                  sg.dev_dxb[dev], sg.dev_lo[dev], sg.dev_hi[dev], sp, gp, C, escape,
                  R_em, Int(nmax), sg.ni, sg.nj, sg.nk, sg.nmb, cnt)
    KernelAbstractions.synchronize(backend)
    img[rng] .= Array(outd)
    return nothing
end

function _gsp_pol_part!(out, rays, rng, sg::DeviceAMRScanGrid, dev, sp, gp, C, escape,
                        R_em, nmax, wg)
    backend = CUDABackend()
    kern = _gsp_pol_amr_kernel!(backend)
    cnt = length(rng)
    y0d = CuArray(rays[rng])
    outd = CuArray{SVector{4,Float64}}(undef, cnt)
    _kern_launch!(kern, wg, cnt, outd, y0d, sg.dev_blk[dev], sg.dev_xlft[dev],
                  sg.dev_dxb[dev], sg.dev_lo[dev], sg.dev_hi[dev], sp, gp, C, escape,
                  R_em, Int(nmax), sg.ni, sg.nj, sg.nk, sg.nmb, cnt)
    KernelAbstractions.synchronize(backend)
    out[rng] .= Array(outd)
    return nothing
end

"""
    march_rays_gpu_spacetime(cfg, st, snap_or_scangrid, rays; polarized=false,
                             devices=select_devices(), nmax=cfg.max_steps, C=80,
                             R_em=Inf, wg=<pol _GPOL_WG / unpol nothing>)
        -> Vector{Float64} (unpol)  |  Matrix{Float64}(4, length(rays)) (polarized)

Fused single-pass GPU march of an arbitrary camera-ray list in an arbitrary stationary
spacetime `st` (`SuperposedKerrSchild`, `ADMGridSpacetime`, or a pre-uploaded
`upload_adm` Dict), sampling the AMR fluid on device through the scan locate. The
geodesic step is the CPU engine in-kernel (bit-compatible with `trace_rays_sampled`).
`polarized=false` returns Stokes I as a flat `Vector{Float64}`; `polarized=true` runs
the full-IQUV reverse-Mueller march and returns the `4 × length(rays)` Stokes matrix
(rows I,Q,U,V). `snap_or_scangrid`: `AthenaKSnapshot` (uploaded internally) or
`DeviceAMRScanGrid` (reused). Float64 only.
"""
function march_rays_gpu_spacetime(cfg, st, snap_or_scangrid, rays::Vector{SVector{8,Float64}};
                                  polarized::Bool = false,
                                  devices = select_devices(), nmax::Int = cfg.max_steps,
                                  C::Real = 80, R_em::Real = Inf, wg = :default)
    wg === :default && (wg = polarized ? _GPOL_WG : nothing)
    sg = snap_or_scangrid isa DeviceAMRScanGrid ? snap_or_scangrid :
         upload_amr_scan(snap_or_scangrid, devices)
    sps = _gsp_per_device(st, devices)
    escape = 1.1 * cfg.camera.distance
    Cf = Float64(C)
    nrays = length(rays)
    nd = length(devices)
    if polarized
        gp = _gpol_params(cfg)
        out = Vector{SVector{4,Float64}}(undef, nrays)
        @sync for di in 1:nd
            idx = di:nd:nrays; dev = devices[di]
            isempty(idx) && continue
            @async begin
                CUDA.device!(dev)
                _gsp_pol_part!(out, rays, idx, sg, dev, sps[dev], gp, Cf, escape,
                               Float64(R_em), nmax, wg)
            end
        end
        M = Matrix{Float64}(undef, 4, nrays)
        @inbounds for j in 1:nrays
            o = out[j]
            M[1, j] = o[1]; M[2, j] = o[2]; M[3, j] = o[3]; M[4, j] = o[4]
        end
        return M
    end
    gp = _gimg_params(cfg)
    img = Vector{Float64}(undef, nrays)
    @sync for di in 1:nd
        idx = di:nd:nrays; dev = devices[di]
        isempty(idx) && continue
        @async begin
            CUDA.device!(dev)
            _gsp_part!(img, rays, idx, sg, dev, sps[dev], gp, Cf, escape,
                       Float64(R_em), nmax, wg)
        end
    end
    return img
end

"""
    render_gpu_spacetime(cfg, st, snap_or_scangrid; polarized=false,
                         devices=select_devices(), nmax=cfg.max_steps, C=80,
                         R_em=Inf, wg=<pol _GPOL_WG / unpol nothing>)
        -> Matrix{Float64} (unpol)  |  NamedTuple (I,Q,U,V) (polarized)

Full-frame generic-spacetime fused GPU render: the device twin of
`render(cfg, snap; spacetime=st, polarized=...)` — camera rays built with the same
`camera_rays(st, cam)` (host CPU metric), marched/sampled/accumulated entirely on
device. `polarized=false` returns the Stokes-I `Matrix{Float64}` in `render().I`
layout; `polarized=true` returns the `(I, Q, U, V)` NamedTuple of npix×npix matrices
(full-IQUV reverse-Mueller march, parallel-transported camera frame).
"""
function render_gpu_spacetime(cfg, st, snap_or_scangrid; polarized::Bool = false,
                              devices = select_devices(),
                              nmax::Int = cfg.max_steps, C::Real = 80, R_em::Real = Inf,
                              wg = :default)
    st_host = st isa Dict ? error("pass the ADMGridSpacetime itself to render_gpu_spacetime") : st
    cam = PlaneParallelCamera(cfg.camera.inclination_deg, cfg.camera.distance,
                              cfg.camera.fov, cfg.camera.resolution)
    rays = camera_rays(st_host, cam)
    npx = cam.npix
    if polarized
        M = march_rays_gpu_spacetime(cfg, st, snap_or_scangrid, rays; polarized = true,
                                     devices = devices, nmax = nmax, C = C,
                                     R_em = R_em, wg = wg)
        return (I = reshape(M[1, :], npx, npx), Q = reshape(M[2, :], npx, npx),
                U = reshape(M[3, :], npx, npx), V = reshape(M[4, :], npx, npx))
    end
    img = march_rays_gpu_spacetime(cfg, st, snap_or_scangrid, rays; devices = devices,
                                   nmax = nmax, C = C, R_em = R_em, wg = wg)
    return reshape(img, npx, npx)
end

"""
    trace_shadow_gpu(st, rays; C=60.0, escape=95.0, nmax=6000,
                     devices=select_devices(), wg=nothing)
        -> (status::Vector{Float64}, nsteps, maxnull, rfinal, yfinal)

No-fluid GPU shadow/capture tracer for an arbitrary stationary spacetime (device twin
of the z4c shadow tracer and the capture/null gates of the superposed tracer).
`status`: 1 = captured, 2 = escaped, 0 = step-limited. `maxnull` is the per-ray
max |g^{μν}p_μp_ν| (CPU `null_error`) over the march.
"""
function trace_shadow_gpu(st, rays::Vector{SVector{8,Float64}};
                          C::Real = 60.0, escape::Real = 95.0, nmax::Int = 6000,
                          devices = select_devices(), wg = nothing)
    sps = _gsp_per_device(st, devices)
    n = length(rays)
    ys = Vector{SVector{8,Float64}}(undef, n)
    ms = Vector{SVector{4,Float64}}(undef, n)
    nd = length(devices)
    @sync for di in 1:nd
        idx = di:nd:n; dev = devices[di]
        isempty(idx) && continue
        @async begin
            CUDA.device!(dev)
            backend = CUDABackend()
            kern = _gsp_shadow_kernel!(backend)
            cnt = length(idx)
            y0d = CuArray(rays[idx])
            oyd = CuArray{SVector{8,Float64}}(undef, cnt)
            omd = CuArray{SVector{4,Float64}}(undef, cnt)
            _kern_launch!(kern, wg, cnt, oyd, omd, y0d, sps[dev], Float64(C),
                          Float64(escape), Int(nmax), cnt)
            KernelAbstractions.synchronize(backend)
            ys[idx] .= Array(oyd)
            ms[idx] .= Array(omd)
        end
    end
    return ([m[1] for m in ms], [m[2] for m in ms], [m[3] for m in ms],
            [m[4] for m in ms], ys)
end

# ---------------------------------------------------------------- device binary orbit
"""
    DevBinaryOrbit(q, times, data, dt)

Store binary orbital data in arrays that can be adapted for GPU use.

`times` contains coordinate times with uniform spacing `dt`.
`data` has shape (18, nt), ordered as center1, center2, spin1, spin2,
velocity1, velocity2, each with three spatial components.

Use data from an already-loaded `BinaryOrbit`, with the time offset
and secondary-spin scaling already applied.
"""
struct DevBinaryOrbit{T,D}
    q::Float64
    times::T
    data::D
    dt::Float64
end

Adapt.@adapt_structure DevBinaryOrbit

"""
    (orbit::DevBinaryOrbit)(t::Real)

Interpolate orbital data at coordinate time `t`, matching `BinaryOrbit`'s
linear interpolation and endpoint extrapolation.

Return `(center=(c1, c2), velocity=(v1, v2), spin=(a1, a2))`.
Each entry is an `SVector{3,Float64}`; spins are `a = J/M` in primary-mass
units. Times outside the tabulated range are allowed.
"""
@inline function (orbit::DevBinaryOrbit)(t::Real)
    time = Float64(t)
    nt = length(orbit.times)

    i0 = trunc(Int, (time - orbit.times[1]) / orbit.dt + 1000) - 1000
    i0 = clamp(i0, 0, nt - 1)
    i = i0 + 1

    left = i == nt ? nt - 1 : i
    right = i == nt ? nt : i + 1
    offset = time - orbit.times[i]

    values = SVector{18,Float64}(ntuple(Val(18)) do k
        slope = (
            orbit.data[k, right] - orbit.data[k, left]
        ) / orbit.dt

        offset * slope + orbit.data[k, i]
    end)

    return (
        center = (
            SVector(values[1], values[2], values[3]),
            SVector(values[4], values[5], values[6]),
        ),
        velocity = (
            SVector(values[13], values[14], values[15]),
            SVector(values[16], values[17], values[18]),
        ),
        spin = (
            SVector(values[7], values[8], values[9]),
            SVector(values[10], values[11], values[12]),
        ),
    )
end


"""
    Adapt.adapt_structure(to, st::SuperposedBoostedKerrSchildCartesian)

Adapt the metric's orbit storage to `to`, preserving the mass ratio.
The orbit type must support adaptation; this method does not itself
upload a CPU `BinaryOrbit` to the GPU.
"""
function Adapt.adapt_structure(
    to, st::SuperposedBoostedKerrSchildCartesian,
)
    return SuperposedBoostedKerrSchildCartesian(
        st.q, Adapt.adapt(to, st.orbit),
    )
end

# ---------------------------------------------------------------- binary image kernel
"""
    _sbks_image_kernel!(image, statuses, steps, rays, st,
                        blk, xlft, dxb, lo, hi,
                        ni, nj, nk, nmb, gp,
                        C, escape, nmax, capture_factor)

Trace one ray per CUDA thread and accumulate unpolarized intensity
from the fixed AMR fluid snapshot.

Use negative affine RK4 steps through the time-dependent binary metric.
Evaluate each segment's radiation coefficients at its camera-side endpoint
and weight its emission by the accumulated foreground transmission.

Write intensity, completed step count, and termination status:
1 = captured, 2 = escaped, 3 = step limit, 4 = invalid calculation.
Failed rays receive `NaN` intensity. Capture uses approximate individual-BH
boundaries; escape requires outward motion beyond `escape`.
"""
function _sbks_image_kernel!(
    image, statuses, steps, rays, st,
    blk, xlft, dxb, lo, hi,
    ni, nj, nk, nmb, gp,
    C, escape, nmax, capture_factor,
)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if idx <= length(rays)
        G = BlackLightPlus.Geodesics
        y = rays[idx]
        r_prev = Inf
        Iinv = 0.0
        trans = 1.0
        count = 0
        status = 3

        for step in 0:nmax
            if !all(isfinite, y)
                status = 4
                break
            end

            x = SVector(y[1], y[2], y[3], y[4])
            ctl = G._sbks_trace_geometry(st, x, capture_factor)

            if !ctl.valid
                status = 4
                break
            elseif ctl.captured
                status = 1
                break
            elseif ctl.R > escape && ctl.R > r_prev
                status = 2
                break
            end

            r_prev = ctl.R
            step == nmax && break

            rhs = G.geodesic_rhs(st, y)
            if !all(isfinite, rhs) || rhs[1] <= 0.0
                status = 4
                break
            end

            spatial_rate = norm(SVector(rhs[2], rhs[3], rhs[4]))
            rate = max(abs(rhs[1]), spatial_rate)
            h = min(ctl.dmin / C, 1.0) / rate

            if !isfinite(h) || h <= 0.0
                status = 4
                break
            end

            ynew = G.rk4_step_fixed(st, y, -h)
            if !all(isfinite, ynew) || ynew[1] >= y[1]
                status = 4
                break
            end

            # Camera-side coefficients over a positive affine interval.
            prim = _gamr_sample_scan(
                blk, xlft, dxb, lo, hi,
                ni, nj, nk, nmb, x[2], x[3], x[4],
            )

            if prim[1] > 0.0 && prim[5] > 0.0 &&
               !(prim[6] == 0.0 && prim[7] == 0.0 && prim[8] == 0.0)

                kcov = SVector(y[5], y[6], y[7], y[8])
                gcov = G.metric(st, x)
                gcon = G.inverse_metric(st, x)

                j_inv, a_inv = _gimg_coeffs_st(
                    prim, kcov, gcov, gcon, gp,
                )

                if !(isfinite(j_inv) && isfinite(a_inv))
                    status = 4
                    break
                end

                ds = h * gp.L_unit / gp.nu_obs
                A, B = _gimg_ab(j_inv, a_inv, ds)
                Iinv += trans * B
                trans *= A

                if !(isfinite(Iinv) && isfinite(trans))
                    status = 4
                    break
                end
            end

            y = ynew
            count += 1
        end

        image[idx] = (status == 1 || status == 2) ?
            Iinv * gp.nu_obs^3 : NaN
        statuses[idx] = status
        steps[idx] = count
    end

    return nothing
end

# ---------------------------------------------------------------- binary host renderer
"""
    render_gpu_spacetime(cfg,
                        st::SuperposedBoostedKerrSchildCartesian,
                        snap_or_scangrid;
                        t_obs, capture_factor, polarized=false,
                        C=80.0, nmax=cfg.max_steps,
                        device=CUDA.deviceid(CUDA.device()),
                        return_status=false)

Render unpolarized radiation through the time-dependent binary metric
on one CUDA device. The fluid snapshot stays fixed along each ray.

Pass a CPU metric containing a `BinaryOrbit` and either an `AthenaKSnapshot`
or a `DeviceAMRScanGrid` uploaded to `device`. Require numerical tracing;
polarized rendering is not supported.

`t_obs` sets the camera time. `capture_factor >= 1` sets approximate
BH capture boundaries, `C` controls distance-based sampling, and `nmax`
limits steps per ray. Escape occurs on outward motion beyond
`1.1 * cfg.camera.distance`. Orbit endpoint extrapolation is allowed.

`cfg.m_bh_cgs` sets the physical primary BH mass. The snapshot's
internal-energy conversion must match `cfg.plasma.gamma`.

Return an intensity matrix with rows in beta and columns in alpha.
With `return_status=true`, return `(I, status, steps)` instead:
1 = captured, 2 = escaped, 3 = step limit, 4 = invalid calculation.
Failed rays have `NaN` intensity; without `return_status`, failures raise
an error.
"""
function render_gpu_spacetime(
    cfg,
    st::SuperposedBoostedKerrSchildCartesian,
    snap_or_scangrid;
    t_obs::Real,
    capture_factor::Float64,
    polarized::Bool=false,
    C::Float64=80.0,
    nmax::Int=cfg.max_steps,
    device::Integer=CUDA.deviceid(CUDA.device()),
    return_status::Bool=false,
)
    cfg.tracer === :numerical ||
        throw(ArgumentError("the binary metric requires tracer=:numerical"))
    polarized &&
        throw(ArgumentError("binary polarized GPU rendering is not implemented"))
    st.orbit isa BinaryOrbit ||
        throw(ArgumentError("pass a CPU metric containing a BinaryOrbit"))
    isfinite(st.q) && st.q > 0.0 ||
        throw(ArgumentError("binary mass ratio must be finite and positive"))
    isfinite(C) && C > 0.0 ||
        throw(ArgumentError("C must be finite and positive"))
    isfinite(capture_factor) && capture_factor >= 1.0 ||
        throw(ArgumentError("capture_factor must be finite and at least 1"))
    0 < nmax <= typemax(Int32) ||
        throw(ArgumentError("nmax must fit in a positive Int32"))
    cfg.camera.resolution > 0 ||
        throw(ArgumentError("camera resolution must be positive"))
    isfinite(cfg.camera.distance) && cfg.camera.distance > 0.0 ||
        throw(ArgumentError("camera distance must be finite and positive"))

    cam = PlaneParallelCamera(
        cfg.camera.inclination_deg,
        cfg.camera.distance,
        cfg.camera.fov,
        cfg.camera.resolution,
    )
    rays = camera_rays(st, cam; t_obs=t_obs)

    dev = Int(device)
    CUDA.device!(dev)

    if snap_or_scangrid isa DeviceAMRScanGrid
        sg = snap_or_scangrid
        for arrays in (
            sg.dev_blk, sg.dev_xlft, sg.dev_dxb, sg.dev_lo, sg.dev_hi,
        )
            haskey(arrays, dev) ||
                throw(ArgumentError("fluid grid is not uploaded to device $dev"))
        end
    elseif snap_or_scangrid isa AthenaKSnapshot
        sg = upload_amr_scan(snap_or_scangrid, [dev])
    else
        throw(ArgumentError("pass an AthenaKSnapshot or DeviceAMRScanGrid"))
    end

    orb = st.orbit
    orb_gpu = DevBinaryOrbit(
        orb.q, CuArray(orb.times), CuArray(orb.data), orb.dt,
    )
    st_gpu = SuperposedBoostedKerrSchildCartesian(st.q, orb_gpu)

    nrays = length(rays)
    rays_gpu = CuArray(rays)
    image_gpu = CUDA.zeros(Float64, nrays)
    status_gpu = CUDA.zeros(Int32, nrays)
    steps_gpu = CUDA.zeros(Int32, nrays)

    gp = _gimg_params(cfg)
    escape = 1.1 * cam.d

    @cuda threads=32 blocks=cld(nrays, 32) _sbks_image_kernel!(
        image_gpu, status_gpu, steps_gpu, rays_gpu, st_gpu,
        sg.dev_blk[dev],
        sg.dev_xlft[dev],
        sg.dev_dxb[dev],
        sg.dev_lo[dev],
        sg.dev_hi[dev],
        sg.ni, sg.nj, sg.nk, sg.nmb,
        gp, C, escape, nmax, capture_factor,
    )
    CUDA.synchronize()

    I = reshape(Array(image_gpu), cam.npix, cam.npix)
    status = reshape(Array(status_gpu), cam.npix, cam.npix)
    steps = reshape(Array(steps_gpu), cam.npix, cam.npix)

    if return_status
        return (I=I, status=status, steps=steps)
    end

    failed = count(s -> s != 1 && s != 2, status)
    failed == 0 ||
        error("$failed GPU rays failed; use return_status=true for diagnostics")
    all(isfinite, I) || error("GPU image contains nonfinite intensities")

    return I
end