# mc_gpu.jl — GPU Monte Carlo radiative transfer (KernelAbstractions + CUDA).
#
# Device port of the sphere Monte Carlo: per superphoton, seed the Philox
# stream `MCRng(seed, i)` (same key as the CPU path), generate a synchrotron
# packet, track it through the flat sphere, and bin it atomically into a
# device (theta, E) spectrum; a layered host loop adds Compton scattering
# (cf. gpumonty scattering.cu). Leaf samplers are the same pure/isbits CPU
# functions; the weight lookup and scatter march use the shared mc_common.jl
# bodies, so CPU and GPU cannot drift. Include after `using BlackLightPlus`.

using StaticArrays
using KernelAbstractions
using CUDA
using LinearAlgebra: transpose

using BlackLightPlus: MCRng, mc_uniform, sample_direction,
                    int_jnu_zone, j_nu, jmax, alpha_nu, emission_tetrad,
                    SphereModel, sphere_zone, WeightTable, build_weight_table,
                    zone_nsuperphotons, Spectrum, merge_spectra!,
                    HotcrossTable, scatter_superphoton, bias_func, ScatterResult
using BlackLightPlus.Constants: ME, CL, HPL

# CPU-side constants, referenced (not re-typed) so the kernels can never drift
# from mc_cpu.jl / track.jl / generate.jl / emission_tables.jl.
const _MCM = BlackLightPlus.MonteCarlo

# ---------------------------------------------------------------- multi-GPU seed stride
# Multi-device split: device `di` (1-based) uses global seed
# `seed + (di-1)*_MC_MULTI_SEED_STRIDE`, so per-device Philox streams never
# overlap; `di == 1` leaves the seed unchanged, so `devices=[d]` reproduces the
# single-device path bit-for-bit. Stride 2^30 keeps the masked 32-bit seeds far apart.
const _MC_MULTI_SEED_STRIDE = 1_073_741_824   # 2^30

# ---------------------------------------------------------------- device selection
if !@isdefined(select_devices)
"""
    select_devices(candidates=[0,1]; util_threshold=15, min_free_gb=8) -> Vector{Int}

Probe each visible CUDA device in `candidates` and return the subset that is
idle enough: free memory >= `min_free_gb` GB and (when NVML is available)
compute utilization <= `util_threshold` %. Always returns at least one device.
Definition is guarded so this file stays self-contained alongside gpu_image.jl.
"""
function select_devices(candidates = [0, 1]; util_threshold::Real = 15, min_free_gb::Real = 8)
    vis = try
        filter(d -> 0 <= d < CUDA.ndevices(), candidates)
    catch
        return collect(candidates)
    end
    isempty(vis) && return collect(candidates)
    idle = Int[]
    for d in vis
        ok = try
            CUDA.device!(d)
            free, _ = CUDA.memory_info()
            u = try
                Float64(CUDA.NVML.utilization_rates(CUDA.NVML.Device(CUDA.uuid(CUDA.CuDevice(d)))).compute) * 100.0
            catch
                -1.0
            end
            (free / 1024^3 >= min_free_gb) && (u < 0 || u <= util_threshold)
        catch
            false
        end
        ok && push!(idle, d)
    end
    return isempty(idle) ? vis[1:1] : idle
end
end # @isdefined select_devices

# ---------------------------------------------------------------- device absorption rate
# Device twin of track.jl `_nu_alpha` for the uniform flat sphere (static fluid,
# B along x1): the invariant absorption rate nu*alpha_nu at `x` for wavevector
# `kcon`, zero outside the sphere.
@inline function _mc_nu_alpha_dev(x::SVector{4}, kcon::SVector{4}, ne, te, B, R)
    r2 = x[2] * x[2] + x[3] * x[3] + x[4] * x[4]
    r2 <= R * R || return 0.0
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    ks <= 0.0 && return 0.0
    cth = kcon[2] / ks
    sth = sqrt(max(0.0, 1.0 - cth * cth))
    nu = kcon[1] * ME * CL * CL / HPL
    return nu * alpha_nu(ne, te, B, nu, sth)
end

# ---------------------------------------------------------------- device straight-line track
# Mirror of track.jl `track_superphoton` for the thin sphere: k^mu is null and
# constant (flat metric), so the geodesic is a straight line and the weight
# decays by the trapezoid-integrated absorption depth each step. Returns only
# the final weight (position/momentum are unchanged for the record).
@inline function _mc_track_thin_w(x0::SVector{4}, pcov::SVector{4}, w0,
                                  gcon::SMatrix{4,4}, ne, te, B, R,
                                  step_frac, track_max::Int)
    kcon = gcon * pcov
    ks = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    dlam = ks > 0.0 ? step_frac * R / ks : R
    Cabs = HPL / (ME * CL * CL)
    na0 = _mc_nu_alpha_dev(x0, kcon, ne, te, B, R)
    w = w0
    x = x0
    @inbounds for _ in 1:track_max
        xn = x .+ kcon .* dlam
        na1 = _mc_nu_alpha_dev(xn, kcon, ne, te, B, R)
        dtau = 0.5 * (na0 + na1) * Cabs * dlam
        w *= exp(-dtau)
        x = xn
        na0 = na1
        r2 = xn[2] * xn[2] + xn[3] * xn[3] + xn[4] * xn[4]
        r2 <= R * R || break
    end
    return w
end

# ---------------------------------------------------------------- generation (shared recipe)
# Frequency-rejection accept numerator, the device mirror of generate.jl
# `_gen_int_jnu`: exact `int_jnu_zone` when `lint === nothing` (default,
# selected by dispatch), or the uploaded `log int_jnu` table read through the
# shared interpolation body when `tabulated_gen` is opted in (`p` then carries
# the grid scalars `lint_dlnu`/`lint_n`; `lnu_min` is shared with the weight grid).
@inline _mc_gen_int_jnu(::Nothing, p, nu) = int_jnu_zone(p.ne, p.te, p.B, nu)
@inline _mc_gen_int_jnu(lint, p, nu) =
    _MCM._mc_weight_at(nu, lint, p.lnu_min, p.lint_dlnu, p.lint_n)

# Device generator, called by the thin and scatter kernels: uniform-ball
# emission point + frequency & pitch rejection -> covariant seed momentum,
# mirroring sphere_model.jl `sample_sphere_photon` + generate.jl
# `sample_zone_photon`. The emission tetrad `p.E` is host-precomputed once per
# run (constant for the uniform flat sphere). `lint` selects the opt-in
# tabulated frequency rejection (`nothing` = exact default; `p.dnmax` must then
# be the matching `tabulated_dnmax` envelope).
@inline function _mc_gen_sphere_photon(rng::MCRng, p, lwgt, lint)
    u, rng = mc_uniform(rng)
    r = p.R * cbrt(u)
    dir, rng = sample_direction(rng)
    x = SVector(0.0, r * dir[1], r * dir[2], r * dir[3])

    nu = 0.0
    w = 0.0
    for _ in 1:p.reject_max
        u1, rng = mc_uniform(rng)
        nu = exp(u1 * (p.lnu_max - p.lnu_min) + p.lnu_min)
        w = _MCM._mc_weight_at(nu, lwgt, p.lnu_min, p.dlnu, p.n_nodes)
        u2, rng = mc_uniform(rng)
        accept = (_mc_gen_int_jnu(lint, p, nu) / (w + p.weight_floor)) / p.dnmax
        u2 <= accept && break
    end

    jm = jmax(p.ne, p.te, p.B, nu)
    cth = 0.0
    for _ in 1:p.reject_max
        u3, rng = mc_uniform(rng)
        cth = 2.0 * u3 - 1.0
        sth0 = sqrt(max(0.0, 1.0 - cth * cth))
        u4, rng = mc_uniform(rng)
        u4 <= j_nu(p.ne, p.te, p.B, nu, sth0) / jm && break
    end
    u5, rng = mc_uniform(rng)
    phi = 2.0 * π * u5
    sth = sqrt(max(0.0, 1.0 - cth * cth))

    eps = nu * HPL / (ME * CL * CL)
    k_tet = SVector(eps, eps * cth, eps * sth * cos(phi), eps * sth * sin(phi))
    kcon = transpose(p.E) * k_tet
    pcov = p.gcov * kcon
    return x, pcov, w, rng
end

# ---------------------------------------------------------------- the per-superphoton kernel
"""
    _mc_thin_kernel!(sumw, sumwE, cnt, lwgt, lint, p, N)

One superphoton per global index `i in 1:N`: seeds `MCRng(p.seed, i)` (the same
stream the CPU path uses for photon `i`), generates a synchrotron packet
(`_mc_gen_sphere_photon`), tracks it straight to escape with absorption-weight
decay, and atomically bins `(count, w, w*E)` by exit angle + energy. `p` is the
isbits parameter bundle; `lwgt` the uploaded weight table; `lint` the optional
tabulated-generation table (`nothing` = exact default).
"""
@kernel function _mc_thin_kernel!(sumw, sumwE, cnt, @Const(lwgt), lint, p, @Const(N))
    i = @index(Global)
    @inbounds if i <= N
        # ---- generate (stream keyed (seed, photon_id)) ----
        x, pcov, w, _ = _mc_gen_sphere_photon(MCRng(p.seed, i), p, lwgt, lint)

        # ---- track straight to escape with absorption-weight decay ----
        w = _mc_track_thin_w(x, pcov, w, p.gcon, p.ne, p.te, p.B, p.R,
                             p.step_frac, p.track_max)

        # ---- record: bin by exit (theta, E) (spectrum.jl record!) ----
        Ee = -pcov[1]                                    # E = -p_t
        if Ee > 0.0
            nu_out = Ee * ME * CL * CL / HPL
            ie = unsafe_trunc(Int, floor((log(nu_out) - p.spec_lnu_min) / p.spec_dlnu)) + 1
            if 1 <= ie <= p.n_ebins
                ps = sqrt(pcov[2] * pcov[2] + pcov[3] * pcov[3] + pcov[4] * pcov[4])
                mu = ps > 0.0 ? abs(pcov[4]) / ps : 0.0  # |cos theta| about z, folded
                thf = acos(clamp(mu, 0.0, 1.0))
                ith = clamp(unsafe_trunc(Int, floor(thf / p.dth)) + 1, 1, p.n_thbins)
                CUDA.@atomic cnt[ith, ie]   += Int64(1)
                CUDA.@atomic sumw[ith, ie]  += w
                CUDA.@atomic sumwE[ith, ie] += w * Ee
            end
        end
    end
end

# ---------------------------------------------------------------- host driver
"""
    mc_spectrum_gpu(model::SphereModel, Ns; device=0, workgroupsize=nothing,
                    seed=0x5eed5eed, nu_min=1e8, nu_max=1e16, n_esamp=400,
                    n_ebins=200, n_thbins=6) -> Spectrum

GPU thin-sphere Monte Carlo, the device twin of `mc_spectrum` (mc_cpu.jl):
builds the same one-zone weight table and target count `N_gen` on the host (so
packet count and Philox streams match the CPU path photon-for-photon), launches
`_mc_thin_kernel!` on `device`, and reduces the atomic device bins into the
returned host `Spectrum`. Pass `devices` for a multi-GPU split: each device
builds the full-`Ns` weight table and runs its `n_gen_override` share with a
distinct Philox seed (splitting `Ns` instead would double every weight — a 2x
total-power bug); `devices=[d]` reproduces the single-device run bit-for-bit.
"""
function mc_spectrum_gpu(model::SphereModel, Ns::Real;
                         device::Integer = 0, devices = nothing, workgroupsize = nothing,
                         seed::Integer = 0x5eed5eed,
                         nu_min::Real = _MCM._MC_NU_MIN_DEFAULT,
                         nu_max::Real = _MCM._MC_NU_MAX_DEFAULT,
                         n_esamp::Integer = _MCM._MC_N_ESAMP_DEFAULT,
                         n_ebins::Integer = 200, n_thbins::Integer = 6,
                         tabulated_gen::Bool = false,
                         n_gen_override::Union{Nothing,Integer} = nothing)
    # ---- multi-GPU split ----
    # The thin spectrum is additive, so the combined result is the per-device
    # sum — but each device must build the full-`Ns` weight table and run only
    # its N_gen/ndev share (splitting `Ns` would double the weights). Devices
    # run concurrently; distinct seeds give disjoint streams.
    if devices !== nothing
        devlist = collect(Int, devices)
        ndev = length(devlist)
        # full-ensemble N_gen (from the FULL Ns) so per-device weights stay full-normalised.
        band = log(nu_max) - log(nu_min)
        zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
        wgt = build_weight_table([zone], Float64(Ns) / band;
                                 n_esamp = n_esamp, nu_min = nu_min, nu_max = nu_max)
        N_target, _ = zone_nsuperphotons(zone, wgt)
        N_gen_full = max(1, round(Int, N_target))
        parts = Vector{Spectrum}(undef, ndev)
        @sync for di in 1:ndev
            dev = devlist[di]
            sdev = Int(seed) + (di - 1) * _MC_MULTI_SEED_STRIDE
            n_this = div(N_gen_full, ndev) + (di <= mod(N_gen_full, ndev) ? 1 : 0)
            @async begin
                CUDA.device!(dev)
                parts[di] = mc_spectrum_gpu(model, Ns; device = dev,
                                            workgroupsize = workgroupsize, seed = sdev,
                                            nu_min = nu_min, nu_max = nu_max,
                                            n_esamp = n_esamp, n_ebins = n_ebins,
                                            n_thbins = n_thbins,
                                            tabulated_gen = tabulated_gen,
                                            n_gen_override = n_this)
            end
        end
        combined = Spectrum(; nu_min = nu_min, nu_max = nu_max,
                            n_ebins = n_ebins, n_thbins = n_thbins)
        for pr in parts
            merge_spectra!(combined, pr)
        end
        return combined
    end

    # ---- host generation tables (identical to mc_spectrum) ----
    band = log(nu_max) - log(nu_min)
    Ns_param = Float64(Ns) / band                        # so N_target ~ Ns
    zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
    wgt = build_weight_table([zone], Ns_param;
                             n_esamp = n_esamp, nu_min = nu_min, nu_max = nu_max)
    N_target, dnmax = zone_nsuperphotons(zone, wgt)
    # full-Ns weight table always; the split overrides ONLY the launched superphoton count.
    N_gen = n_gen_override === nothing ? max(1, round(Int, N_target)) : max(1, Int(n_gen_override))
    # opt-in tabulated generation (host tables; the CPU opt-in's exact mirror).
    gt = tabulated_gen ? _MCM.build_int_jnu_table(zone, wgt) : nothing
    tabulated_gen && (dnmax = _MCM.tabulated_dnmax(gt, wgt))

    # emission tetrad is position-independent for the flat, uniform sphere -> host-constant.
    Etet = emission_tetrad(model.ucon, model.ucov, model.bcon, model.gcov, model.gcon)

    # host spectrum: carries the grid scalars AND receives the reduced device bins.
    spec = Spectrum(; nu_min = nu_min, nu_max = nu_max,
                    n_ebins = n_ebins, n_thbins = n_thbins)

    # ---- device launch ----
    CUDA.device!(Int(device))
    backend = CUDABackend()
    lwgt_d = CuArray(wgt.lwgt)
    lint_d = gt === nothing ? nothing : CuArray(gt.ljnu)
    d_sumw  = CUDA.zeros(Float64, n_thbins, n_ebins)
    d_sumwE = CUDA.zeros(Float64, n_thbins, n_ebins)
    d_cnt   = CUDA.zeros(Int64,   n_thbins, n_ebins)

    p = (seed = UInt32(Int(seed) & 0xffffffff),
         ne = model.n_e, te = model.theta_e, B = model.B, R = model.R,
         lnu_min = wgt.lnu_min, lnu_max = wgt.lnu_max, dlnu = wgt.dlnu,
         n_nodes = length(wgt.lwgt), dnmax = dnmax,
         weight_floor = _MCM._WEIGHT_FLOOR, reject_max = _MCM._MC_REJECT_MAX_ITER,
         spec_lnu_min = spec.lnu_min, spec_dlnu = spec.dlnu,
         n_ebins = Int(n_ebins), n_thbins = Int(n_thbins),
         dth = (0.5 * π) / n_thbins,
         step_frac = _MCM._MC_STEP_FRAC, track_max = _MCM._MC_TRACK_MAX_STEPS,
         E = Etet, gcov = model.gcov, gcon = model.gcon)
    gt === nothing || (p = merge(p, (lint_dlnu = gt.dlnu, lint_n = length(gt.ljnu))))

    kern = _mc_thin_kernel!(backend)
    if workgroupsize === nothing
        kern(d_sumw, d_sumwE, d_cnt, lwgt_d, lint_d, p, N_gen; ndrange = N_gen)
    else
        kern(d_sumw, d_sumwE, d_cnt, lwgt_d, lint_d, p, N_gen;
             ndrange = N_gen, workgroupsize = min(Int(workgroupsize), N_gen))
    end
    KernelAbstractions.synchronize(backend)

    # ---- reduce device bins into the host Spectrum ----
    spec.sumw  .= Array(d_sumw)
    spec.sumwE .= Array(d_sumwE)
    spec.count .= Array(d_cnt)
    return spec
end

# ======================================================================================
# GPU COMPTON SCATTERING (idealized Comptonization sphere)
#
# Layered, not recursive (after gpumonty scattering.cu `scattering_flow_control`):
# a host loop over scattering layers. The primary kernel marches the seed
# photons with the biased exponential-transform scatter trigger; on a trigger it
# atomically appends the Compton-scattered child to a preallocated device SoA
# buffer via a global slot counter and continues the depleted incident beam;
# escaped photons are binned into the per-order spectrum. The host then launches
# the same march kernel on each layer's buffered children (ping-pong buffers)
# for orders 1..max_scatt. Unbiasedness: each child carries
# w(1-e^{-dtau_s})/(1-e^{-b dtau_s}) at spawn probability 1-exp(-b*dtau_s), so
# E[spawned weight] equals the physical scatter loss; the device accumulates
# both totals (striped atomics) and their ratio must be ~1 independent of the
# bias. The electron / Klein-Nishina rejection loops run in-kernel (pure,
# isbits, capped); the scatter sampler sits behind a `@noinline` boundary so it
# does not inflate the march's register footprint.

# Diagnostic-accumulator stripe count: threads add their per-photon totals to
# diag[:, 1 + (id-1)%_MC_DIAG_STRIPE], summed on the host — avoids single-address atomic
# serialisation across ~N_gen threads.
const _MC_DIAG_STRIPE = 64

# ---------------------------------------------------------------- device scatter opacity
# Device twin of track.jl `_nu_alpha_scatt` reading the uploaded hot-cross table `logsig`
# (a plain `CuArray`): the invariant scattering rate nu*alpha_s = nu*n_e*sigma_h(eps,Te) at
# `x` for wavevector `kcon`, zero outside the sphere. eps = HPL*nu/(ME CL^2) = kcon[1].
@inline function _mc_nu_alpha_scatt_dev(x::SVector{4}, kcon::SVector{4}, logsig, ne, te, R)
    r2 = x[2] * x[2] + x[3] * x[3] + x[4] * x[4]
    r2 <= R * R || return 0.0
    nu = kcon[1] * ME * CL * CL / HPL
    eps = HPL * nu / (ME * CL * CL)                      # == kcon[1]
    return nu * ne * _MCM._hotcross_lookup(logsig, eps, te)
end

# ---------------------------------------------------------------- the scatter kick (split)
# The full Compton sampler — electron draw + KN/Thomson rejection loops — behind
# a `@noinline` boundary so its registers/divergence stay out of the lean march.
# Reuses `scatter_superphoton` verbatim (emission tetrad rebuilt in-kernel; the
# spatial axes cancel).
@noinline function _mc_scatter_mom(pcov::SVector{4}, ucon::SVector{4}, ucov::SVector{4},
                                   bcon::SVector{4}, te,
                                   gcov::SMatrix{4,4}, gcon::SMatrix{4,4}, rng::MCRng)
    p_new, _, rng = scatter_superphoton(pcov, ucon, ucov, bcon, te, gcov, gcon, rng)
    return p_new, rng
end

# ---------------------------------------------------------------- record one escapee
# Atomic bin (theta, E) of an escaped photon with covariant momentum `pcov` and weight `w`
# into the order's device spectrum — a device twin of spectrum.jl `record!` on the SPEC grid.
@inline function _mc_record_dev!(sumw, sumwE, cnt, pcov::SVector{4}, w, p)
    Ee = -pcov[1]
    if Ee > 0.0
        nu_out = Ee * ME * CL * CL / HPL
        ie = unsafe_trunc(Int, floor((log(nu_out) - p.spec_lnu_min) / p.spec_dlnu)) + 1
        if 1 <= ie <= p.n_ebins
            ps = sqrt(pcov[2] * pcov[2] + pcov[3] * pcov[3] + pcov[4] * pcov[4])
            mu = ps > 0.0 ? abs(pcov[4]) / ps : 0.0
            thf = acos(clamp(mu, 0.0, 1.0))
            ith = clamp(unsafe_trunc(Int, floor(thf / p.dth)) + 1, 1, p.n_thbins)
            CUDA.@atomic cnt[ith, ie]   += Int64(1)
            CUDA.@atomic sumw[ith, ie]  += w
            CUDA.@atomic sumwE[ith, ie] += w * Ee
        end
    end
    return nothing
end

# ---------------------------------------------------------------- the biased scatter march
# Transport one superphoton straight through the sphere with the biased
# exponential-transform scatter decision: the march is the shared
# `_MCM._mc_march_scatter_core` (mc_common.jl); this wrapper supplies the device
# opacity lookup (uploaded hot-cross table) and the atomic SoA child sink
# (`xb,pb,wb,nb` + slot counter `ctr`). Absorption is neglected (idealized
# Comptonization sphere), matching the CPU path. Returns
# (recorded, w_final, scatter_loss, child_w_local).
@inline function _mc_march_scatter!(x0::SVector{4}, pcov::SVector{4}, w0, nscatt::Int32,
                                    rng::MCRng, p, logsig, xb, pb, wb, nb, ctr)
    nu_alpha_s = (x, kcon) -> _mc_nu_alpha_scatt_dev(x, kcon, logsig, p.ne, p.te, p.R)
    spawn! = (xev, weight_scat, tau_s, srng) -> begin
        p_new, srng = _mc_scatter_mom(pcov, p.ucon, p.ucov, p.bcon, p.te,
                                      p.gcov, p.gcon, srng)
        slot = CUDA.atomic_add!(pointer(ctr, 1), Int64(1))   # 0-based old index
        kept = 0.0
        if slot < p.cap
            j = Int(slot) + 1
            @inbounds begin
                xb[j] = xev
                pb[j] = p_new
                wb[j] = weight_scat
                nb[j] = nscatt + Int32(1)
            end
            kept = weight_scat   # ledger only TRACKED children (overflow drops excluded)
        end
        return kept, srng
    end
    recorded, _, w, _, sloss, child_w, _ =
        _MCM._mc_march_scatter_core(x0, pcov, w0, 0.0, nscatt, rng, p,
                                    nu_alpha_s, spawn!)
    return (recorded, w, sloss, child_w)
end

# ---------------------------------------------------------------- layer-0 (primary) kernel
"""
    _mc_scatter_primary_kernel!(sumw, sumwE, cnt, xb, pb, wb, nb, ctr, diag,
                                lwgt, lint, logsig, p, N)

One seed superphoton per index `i in 1:N`: generate from the sphere (stream
`MCRng(p.seed, i)`), march with the biased scatter trigger (appending order-1
children to the SoA buffer), record the escaped photon into the order-0
spectrum. Per-thread totals (primary weight / scatter loss / order-1 child
weight) reduce into the striped `diag`.
"""
@kernel function _mc_scatter_primary_kernel!(sumw, sumwE, cnt, xb, pb, wb, nb, ctr, diag,
                                             @Const(lwgt), lint, @Const(logsig), p, @Const(N))
    i = @index(Global)
    @inbounds if i <= N
        rng = MCRng(p.seed, i)
        x, pcov, w0, rng = _mc_gen_sphere_photon(rng, p, lwgt, lint)
        recorded, wf, sloss, child_w =
            _mc_march_scatter!(x, pcov, w0, Int32(0), rng, p, logsig, xb, pb, wb, nb, ctr)
        s = (i - 1) % _MC_DIAG_STRIPE + 1
        CUDA.@atomic diag[1, s] += child_w
        CUDA.@atomic diag[2, s] += sloss
        CUDA.@atomic diag[3, s] += w0
        recorded && _mc_record_dev!(sumw, sumwE, cnt, pcov, wf, p)
    end
end

# ---------------------------------------------------------------- layer-n (children) kernel
"""
    _mc_scatter_layer_kernel!(sumw, sumwE, cnt, xin, pin, win, nin,
                              xout, pout, wout, nout, ctr, diag, logsig, p, child_seed, M)

Process `M` buffered scattered photons (order `n`) from the previous layer:
give each a fresh stream `MCRng(child_seed, j)`, march with the biased trigger
(appending order-(n+1) children to the OUT buffer), record the escaped photon
into the order-`n` spectrum. Diagnostic totals reduce into the striped `diag`.
The top layer (n == max_scatt) spawns nothing but still depletes + records,
mirroring the CPU path.
"""
@kernel function _mc_scatter_layer_kernel!(sumw, sumwE, cnt,
                                           @Const(xin), @Const(pin), @Const(win), @Const(nin),
                                           xout, pout, wout, nout, ctr, diag,
                                           @Const(logsig), p, @Const(child_seed), @Const(M))
    j = @index(Global)
    @inbounds if j <= M
        rng = MCRng(child_seed, j)
        x = xin[j]
        pcov = pin[j]
        w0 = win[j]
        nscatt = nin[j]
        recorded, wf, sloss, child_w =
            _mc_march_scatter!(x, pcov, w0, nscatt, rng, p, logsig, xout, pout, wout, nout, ctr)
        s = (j - 1) % _MC_DIAG_STRIPE + 1
        CUDA.@atomic diag[1, s] += child_w
        CUDA.@atomic diag[2, s] += sloss
        recorded && _mc_record_dev!(sumw, sumwE, cnt, pcov, wf, p)
    end
end

# ---------------------------------------------------------------- multi-device reduction
# Sum per-device `ScatterResult`s into one: per-order spectra add, the total
# re-sums the merged orders, and every scalar diagnostic is the device sum.
# Unbiasedness (w_children ~ scatter_loss) is preserved because both numerator
# and denominator add. Mirrors `mc_spectrum_scatter`'s per-chunk reduction.
function _merge_scatter_results(parts::Vector{ScatterResult};
                                spec_nu_min, spec_nu_max, n_ebins, n_thbins)
    K = parts[1].max_scatt
    orders = [Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                       n_ebins = n_ebins, n_thbins = n_thbins) for _ in 0:K]
    for r in parts, n in 1:(K + 1)
        merge_spectra!(orders[n], r.orders[n])
    end
    total = Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                     n_ebins = n_ebins, n_thbins = n_thbins)
    for n in 1:(K + 1)
        merge_spectra!(total, orders[n])
    end
    return ScatterResult(orders, total, K,
                         sum(r.n_primary for r in parts),
                         sum(r.n_children for r in parts),
                         sum(r.w_primary for r in parts),
                         sum(r.w_children for r in parts),
                         sum(r.w_first_order for r in parts),
                         sum(r.scatter_loss for r in parts))
end

# ---------------------------------------------------------------- host driver (scatter)
"""
    mc_spectrum_gpu(model::SphereModel, table::HotcrossTable, Ns; device=0,
                    workgroupsize=nothing, seed=0x5ca77e5,
                    gen_nu_min=1e8, gen_nu_max=1e13, spec_nu_min=1e8, spec_nu_max=1e17,
                    n_esamp=400, n_ebins=140, n_thbins=6, max_scatt=3,
                    capacity_factor=0.5) -> ScatterResult

GPU Compton-scattering Monte Carlo, the device twin of `mc_spectrum_scatter`
(mc_cpu.jl), selected by the extra positional hot-cross `table`: builds the
same synchrotron weight table + seed count `N_gen`, then runs the layered
scatter loop (primary kernel over the seeds, `_mc_scatter_layer_kernel!` over
each layer's buffered children for orders 1..max_scatt), returning a
`ScatterResult` directly comparable to the CPU result. `capacity_factor` sizes
the per-layer child buffer (overflow guarded + warned). Pass `devices` for a
multi-GPU split: each device builds the full-`Ns` weight table and runs its
seed share with a distinct global (hence child) seed — the split fixes the seed
count, never `Ns`, which would double every weight (a 2x total-power bug).
"""
function mc_spectrum_gpu(model::SphereModel, table::HotcrossTable, Ns::Real;
                         device::Integer = 0, devices = nothing, workgroupsize = nothing,
                         seed::Integer = 0x5ca77e5,
                         gen_nu_min::Real = 1.0e8, gen_nu_max::Real = 1.0e13,
                         spec_nu_min::Real = 1.0e8, spec_nu_max::Real = 1.0e17,
                         n_esamp::Integer = _MCM._MC_N_ESAMP_DEFAULT,
                         n_ebins::Integer = 140, n_thbins::Integer = 6,
                         max_scatt::Integer = 3, capacity_factor::Real = 0.5,
                         tabulated_gen::Bool = false,
                         n_gen_override::Union{Nothing,Integer} = nothing)
    # ---- multi-GPU split ----
    # Each device runs the full layered scatter loop on its share of the N_gen
    # seeds, with the full-`Ns` weight table and a distinct global seed (hence
    # distinct child seeds); `_merge_scatter_results` sums per-order spectra +
    # diagnostics. Only the seed count is split, never `Ns` (weight doubling).
    if devices !== nothing
        devlist = collect(Int, devices)
        ndev = length(devlist)
        # full-ensemble N_gen (from the FULL Ns) so per-device weights stay full-normalised.
        zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
        wgt = build_weight_table([zone], Float64(Ns) / (log(gen_nu_max) - log(gen_nu_min));
                                 n_esamp = n_esamp, nu_min = gen_nu_min, nu_max = gen_nu_max)
        N_target, _ = zone_nsuperphotons(zone, wgt)
        N_gen_full = max(1, round(Int, N_target))
        parts = Vector{ScatterResult}(undef, ndev)
        @sync for di in 1:ndev
            dev = devlist[di]
            sdev = Int(seed) + (di - 1) * _MC_MULTI_SEED_STRIDE
            n_this = div(N_gen_full, ndev) + (di <= mod(N_gen_full, ndev) ? 1 : 0)
            @async begin
                CUDA.device!(dev)
                parts[di] = mc_spectrum_gpu(model, table, Ns; device = dev,
                                            workgroupsize = workgroupsize, seed = sdev,
                                            gen_nu_min = gen_nu_min, gen_nu_max = gen_nu_max,
                                            spec_nu_min = spec_nu_min, spec_nu_max = spec_nu_max,
                                            n_esamp = n_esamp, n_ebins = n_ebins,
                                            n_thbins = n_thbins, max_scatt = max_scatt,
                                            capacity_factor = capacity_factor,
                                            tabulated_gen = tabulated_gen,
                                            n_gen_override = n_this)
            end
        end
        return _merge_scatter_results(collect(parts); spec_nu_min = spec_nu_min,
                                      spec_nu_max = spec_nu_max, n_ebins = n_ebins,
                                      n_thbins = n_thbins)
    end

    # ---- host generation tables (identical to mc_spectrum_scatter) ----
    band = log(gen_nu_max) - log(gen_nu_min)
    Ns_param = Float64(Ns) / band
    zone = sphere_zone(model, SVector(0.0, 0.0, 0.0, 0.0))
    wgt = build_weight_table([zone], Ns_param;
                             n_esamp = n_esamp, nu_min = gen_nu_min, nu_max = gen_nu_max)
    N_target, dnmax = zone_nsuperphotons(zone, wgt)
    # full-Ns weight table always; the split overrides ONLY the launched SEED count.
    N_gen = n_gen_override === nothing ? max(1, round(Int, N_target)) : max(1, Int(n_gen_override))
    # opt-in tabulated generation (host tables; the CPU opt-in's exact mirror).
    gt = tabulated_gen ? _MCM.build_int_jnu_table(zone, wgt) : nothing
    tabulated_gen && (dnmax = _MCM.tabulated_dnmax(gt, wgt))
    K = Int(max_scatt)
    Etet = emission_tetrad(model.ucon, model.ucov, model.bcon, model.gcov, model.gcon)

    # spec-grid scalars (a host Spectrum both carries the grid and receives the bins)
    specg = Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                     n_ebins = n_ebins, n_thbins = n_thbins)

    CUDA.device!(Int(device))
    backend = CUDABackend()
    lwgt_d  = CuArray(wgt.lwgt)
    lint_d  = gt === nothing ? nothing : CuArray(gt.ljnu)
    logsig_d = CuArray(table.logsigma)

    # per-order device spectra (orders 0..K)
    d_sumw  = [CUDA.zeros(Float64, n_thbins, n_ebins) for _ in 0:K]
    d_sumwE = [CUDA.zeros(Float64, n_thbins, n_ebins) for _ in 0:K]
    d_cnt   = [CUDA.zeros(Int64,   n_thbins, n_ebins) for _ in 0:K]

    # ping-pong SoA child buffers + slot counter + striped diagnostics
    cap = max(4096, ceil(Int, Float64(capacity_factor) * N_gen))
    xA = CuArray{SVector{4,Float64}}(undef, cap); pA = CuArray{SVector{4,Float64}}(undef, cap)
    wA = CuArray{Float64}(undef, cap);            nA = CuArray{Int32}(undef, cap)
    xB = CuArray{SVector{4,Float64}}(undef, cap); pB = CuArray{SVector{4,Float64}}(undef, cap)
    wB = CuArray{Float64}(undef, cap);            nB = CuArray{Int32}(undef, cap)
    ctr  = CUDA.zeros(Int64, 1)
    diag = CUDA.zeros(Float64, 3, _MC_DIAG_STRIPE)   # [child_w; scatter_loss; primary_w]

    p = (seed = UInt32(Int(seed) & 0xffffffff),
         ne = model.n_e, te = model.theta_e, B = model.B, R = model.R,
         lnu_min = wgt.lnu_min, lnu_max = wgt.lnu_max, dlnu = wgt.dlnu,
         n_nodes = length(wgt.lwgt), dnmax = dnmax,
         weight_floor = _MCM._WEIGHT_FLOOR, reject_max = _MCM._MC_REJECT_MAX_ITER,
         spec_lnu_min = specg.lnu_min, spec_dlnu = specg.dlnu,
         n_ebins = Int(n_ebins), n_thbins = Int(n_thbins), dth = (0.5 * π) / n_thbins,
         step_frac = _MCM._MC_STEP_FRAC, track_max = _MCM._MC_TRACK_MAX_STEPS,
         weight_min = _MCM._MC_WEIGHT_MIN, max_scatt = K, cap = cap,
         E = Etet, gcov = model.gcov, gcon = model.gcon,
         ucon = model.ucon, ucov = model.ucov, bcon = model.bcon)
    gt === nothing || (p = merge(p, (lint_dlnu = gt.dlnu, lint_n = length(gt.ljnu))))

    base_child_seed = (Int(seed) ⊻ 0x0C0FFEE5) & 0xffffffff

    # ---- layer 0: primaries -> order-0 spectrum, order-1 children into (xA,...) ----
    kern0 = _mc_scatter_primary_kernel!(backend)
    if workgroupsize === nothing
        kern0(d_sumw[1], d_sumwE[1], d_cnt[1], xA, pA, wA, nA, ctr, diag,
              lwgt_d, lint_d, logsig_d, p, N_gen; ndrange = N_gen)
    else
        kern0(d_sumw[1], d_sumwE[1], d_cnt[1], xA, pA, wA, nA, ctr, diag,
              lwgt_d, lint_d, logsig_d, p, N_gen; ndrange = N_gen,
              workgroupsize = min(Int(workgroupsize), N_gen))
    end
    KernelAbstractions.synchronize(backend)
    n_cur = Int(Array(ctr)[1])
    n_overflow = max(0, n_cur - cap)
    n_cur = min(n_cur, cap)
    w_first_order = sum(Array(diag)[1, :])   # order-1 child weight (spawned by primaries)
    n_children_total = n_cur

    # ---- layers 1..K: drain the children (ping-pong) ----
    curx, curp, curw, curn = xA, pA, wA, nA
    nxtx, nxtp, nxtw, nxtn = xB, pB, wB, nB
    kernL = _mc_scatter_layer_kernel!(backend)
    for layer in 1:K
        n_cur > 0 || break
        M = n_cur
        CUDA.fill!(ctr, 0)
        cseed = UInt32((base_child_seed ⊻ (0x51ED9E37 * layer)) & 0xffffffff)
        if workgroupsize === nothing
            kernL(d_sumw[layer + 1], d_sumwE[layer + 1], d_cnt[layer + 1],
                  curx, curp, curw, curn, nxtx, nxtp, nxtw, nxtn, ctr, diag,
                  logsig_d, p, cseed, M; ndrange = M)
        else
            kernL(d_sumw[layer + 1], d_sumwE[layer + 1], d_cnt[layer + 1],
                  curx, curp, curw, curn, nxtx, nxtp, nxtw, nxtn, ctr, diag,
                  logsig_d, p, cseed, M; ndrange = M,
                  workgroupsize = min(Int(workgroupsize), M))
        end
        KernelAbstractions.synchronize(backend)
        n_next = Int(Array(ctr)[1])
        n_overflow += max(0, n_next - cap)
        n_next = min(n_next, cap)
        n_children_total += n_next
        n_cur = n_next
        curx, nxtx = nxtx, curx; curp, nxtp = nxtp, curp
        curw, nxtw = nxtw, curw; curn, nxtn = nxtn, curn
    end
    n_overflow > 0 && @warn "mc_spectrum_gpu scatter buffer overflow: $n_overflow children dropped; raise capacity_factor" cap

    # ---- reduce device bins -> host per-order Spectra + total ----
    function _host_spec(idx)
        s = Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                     n_ebins = n_ebins, n_thbins = n_thbins)
        s.sumw  .= Array(d_sumw[idx])
        s.sumwE .= Array(d_sumwE[idx])
        s.count .= Array(d_cnt[idx])
        return s
    end
    orders = [_host_spec(n + 1) for n in 0:K]
    total = Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max,
                     n_ebins = n_ebins, n_thbins = n_thbins)
    for o in orders
        merge_spectra!(total, o)
    end

    dh = Array(diag)
    w_children  = sum(dh[1, :])
    scatter_loss = sum(dh[2, :])
    w_primary   = sum(dh[3, :])
    return ScatterResult(orders, total, K, N_gen, n_children_total,
                         w_primary, w_children, w_first_order, scatter_loss)
end

# ======================================================================================
# GPU BINARY GRMHD MONTE CARLO (time-dependent superposed boosted Kerr-Schild binary)
#
# Device twin of the binary `mc_spectrum_grmhd` (mc_cpu.jl): seed packets are
# generated on the host with the CPU recipe (same zones, weight table, counts and
# Philox streams, so the packet list is identical to the CPU path), uploaded, and
# tracked on the device through the binary metric (`DevBinaryOrbit`) and the
# scan-located AMR fluid (`_gamr_sample_scan`, gpu_spacetime.jl). The fluid decode
# reuses the GPU imager's `_gimg_pfv` / `_gimg_theta_e`, so the device Monte Carlo
# sees the same plasma as the GPU images; absorption and the Compton kernel reuse
# the CPU functions (`_grmhd_nu_alpha`, `_mc_scatter_event`, `scatter_superphoton`,
# `_hotcross_lookup`). Scattering is layered as in the sphere GPU path: the
# primary kernel appends Compton children to a device buffer (atomic slot
# counter); the host relaunches the march on each order's children.

isdefined(@__MODULE__, :_gamr_sample_scan) ||
    Base.include(@__MODULE__, joinpath(@__DIR__, "..", "gpu_spacetime.jl"))

using BlackLightPlus: SuperposedBoostedKerrSchildCartesian, BinaryOrbit, GRMHDModel
const _MCB_G = BlackLightPlus.Geodesics
const _MCB_MP = BlackLightPlus.Constants.MP

# Host -> backend array (Array on CPU(), device array on a GPU backend).
function _mcb_to(backend, A::AbstractArray)
    B = KernelAbstractions.allocate(backend, eltype(A), size(A))
    copyto!(B, A)
    return B
end

# Emitting-plasma state at x on the device: the CPU `plasma_at` decode with the
# GPU imager's fluid-decode bodies. Vacuum (inside = false) outside the grid, for
# non-positive density/pressure, zero field, b^2 <= 0, or a sigma / Theta_e cut.
@inline function _mcb_plasma_at(st, x::SVector{4,Float64}, fl, gp)
    gcov = _MCB_G.metric(st, x)
    gcon = _MCB_G.inverse_metric(st, x)
    z4 = zero(SVector{4,Float64})
    vac = (n_e = 0.0, theta_e = 0.0, B = 0.0, ucon = z4, ucov = z4, bcon = z4,
           gcov = gcov, gcon = gcon, inside = false)
    prim = _gamr_sample_scan(fl.blk, fl.xlft, fl.dxb, fl.lo, fl.hi,
                             fl.ni, fl.nj, fl.nk, fl.nmb, x[2], x[3], x[4])
    rho = prim[1]
    rho <= 0.0 && return vac
    (prim[6] == 0.0 && prim[7] == 0.0 && prim[8] == 0.0) && return vac
    pgas = (gp.gamma - 1.0) * prim[5]
    pgas <= 0.0 && return vac
    ucon, ucov, bcon, b_sq = _gimg_pfv(prim[2], prim[3], prim[4],
                                       prim[6], prim[7], prim[8], gcov, gcon)
    b_sq <= 0.0 && return vac
    rho_cgs = rho * gp.d_unit
    pgas_cgs = pgas * gp.e_unit
    n_e = rho_cgs / (gp.mu * _MCB_MP) / (1.0 + 1.0 / gp.ne_ni)
    bb = sqrt(b_sq) * gp.b_unit
    sigma = b_sq / rho
    beta_inv = b_sq / (2.0 * pgas)
    theta_e = _gimg_theta_e(rho_cgs, pgas_cgs, beta_inv, gp)
    (gp.sigma_cut >= 0.0 && sigma > gp.sigma_cut) && return vac
    (gp.theta_e_min >= 0.0 && theta_e < gp.theta_e_min) && return vac
    return (n_e = n_e, theta_e = theta_e, B = bb, ucon = ucon, ucov = ucov, bcon = bcon,
            gcov = gcov, gcon = gcon, inside = true)
end

# Device scattering rate nu * alpha_s (CPU `_grmhd_nu_alpha_scatt` with the uploaded
# hot-cross table `logsig`).
@inline function _mcb_nu_alpha_scatt(logsig, pl, p::SVector{4})
    pl.inside || return 0.0
    kdotu = dot(p, pl.ucon)
    kdotu >= 0.0 && return 0.0
    nu = -kdotu * ME * CL * CL / HPL
    eps = HPL * nu / (ME * CL * CL)
    return nu * pl.n_e * _MCM._hotcross_lookup(logsig, eps, pl.theta_e)
end

# Device binary geometry (CPU `_grmhd_binary_geometry` without `error`):
# status 0 = step, 1 = captured, 4 = invalid.
@inline function _mcb_geometry(st, y::SVector{8,Float64}, cf::Float64, C::Float64)
    x = SVector(y[1], y[2], y[3], y[4])
    ctl = _MCB_G._sbks_trace_geometry(st, x, cf)
    ctl.valid || return (4, 0.0, 0.0)
    ctl.captured && return (1, ctl.R, 0.0)
    rhs = _MCB_G.geodesic_rhs(st, y)
    rhs[1] > 0.0 || return (4, ctl.R, 0.0)
    rate = max(abs(rhs[1]), sqrt(rhs[2] * rhs[2] + rhs[3] * rhs[3] + rhs[4] * rhs[4]))
    h = min(ctl.dmin / C, _MCM._GRMHD_BINARY_MAX_COORD_STEP) / rate
    (isfinite(h) && h > 0.0) || return (4, ctl.R, 0.0)
    return (0, ctl.R, h)
end

# Atomic record of an escaped packet (CPU `_grmhd_record_binary!` + `record_energy!`).
@inline function _mcb_record!(acc, st, x::SVector{4,Float64}, p::SVector{4,Float64}, w, q)
    E = -p[1]
    E > 0.0 || return nothing
    nu = E * ME * CL * CL / HPL
    ie = unsafe_trunc(Int, floor((log(nu) - q.lnu_min) / q.dlnu)) + 1
    (ie < 1 || ie > q.n_ebins) && return nothing
    kcon = _MCB_G.inverse_metric(st, x) * p
    ksp = sqrt(kcon[2] * kcon[2] + kcon[3] * kcon[3] + kcon[4] * kcon[4])
    ksp <= 0.0 && return nothing
    th = acos(clamp(kcon[4] / ksp, -1.0, 1.0))
    dphi = 2.0 * π / q.n_phi
    j = min(q.n_phi, unsafe_trunc(Int, floor(mod(atan(kcon[3], kcon[2]) + 0.5 * dphi, 2.0 * π) / dphi)) + 1)
    KernelAbstractions.@atomic acc.tot_cnt[ie] += Int64(1)
    KernelAbstractions.@atomic acc.tot_w[ie] += w
    KernelAbstractions.@atomic acc.tot_wE[ie] += w * E
    @inbounds for b in 1:q.n_bands
        if th >= acc.lo[b] && th <= acc.hi[b]
            KernelAbstractions.@atomic acc.cnt[b, j, ie] += Int64(1)
            KernelAbstractions.@atomic acc.sw[b, j, ie] += w
            KernelAbstractions.@atomic acc.swE[b, j, ie] += w * E
        end
    end
    return nothing
end

# One packet's march (device twin of the binary CPU trackers). With
# `q.scattering`, the bounded-bias Compton trigger appends children (order
# nscatt + 1) to `out` through the atomic slot counter, and packets beyond
# `q.tau_abs_max` are dropped. Escaped packets are recorded (power in acc.esc[1]);
# step-limit stops and absorption drops are counted with their power, as on the CPU.
@inline function _mcb_march!(acc, out, st, fl, gp, logsig, q,
                             x0::SVector{4,Float64}, p0::SVector{4,Float64}, w0::Float64,
                             tau_a0::Float64, tau_s0::Float64, nscatt::Int32, rng::MCRng)
    y = SVector(x0[1], x0[2], x0[3], x0[4], p0[1], p0[2], p0[3], p0[4])
    x = x0; p = p0; w = w0; tau_a = tau_a0; tau_s = tau_s0
    Cgr = q.L_unit * HPL / (ME * CL * CL)
    pl = _mcb_plasma_at(st, x, fl, gp)
    na0 = _MCM._grmhd_nu_alpha(pl, p)
    ns0 = q.scattering ? _mcb_nu_alpha_scatt(logsig, pl, p) : 0.0
    r_prev = Inf
    status = 3
    nspawn = 0
    for _ in 1:q.max_steps
        s, r, h = _mcb_geometry(st, y, q.cf, q.C)
        if s != 0
            status = s
            break
        end
        if r > q.R_record || (r > q.Rem && r > r_prev)
            status = 2
            break
        end
        r_prev = r
        y2 = _MCB_G.rk4_step_fixed(st, y, h)
        xn = SVector(y2[1], y2[2], y2[3], y2[4])
        pn = SVector(y2[5], y2[6], y2[7], y2[8])
        pln = _mcb_plasma_at(st, xn, fl, gp)
        na1 = _MCM._grmhd_nu_alpha(pln, pn)
        dtau_a = 0.5 * (na0 + na1) * Cgr * h
        dtau_s = 0.0
        if q.scattering
            ns1 = _mcb_nu_alpha_scatt(logsig, pln, pn)
            dtau_s = 0.5 * (ns0 + ns1) * Cgr * h
            if dtau_s > 0.0 && pln.inside && Int(nscatt) < q.max_scatt
                b = _MCM.grmhd_bias(q.bias0, pln.theta_e, q.theta_e2_mean, tau_s, q.bias_K)
                spawn, weight_scat, frac, rng =
                    _MCM._mc_scatter_event(w, b, dtau_s, _MCM._MC_WEIGHT_MIN, rng)
                if spawn
                    yev = _MCB_G.rk4_step_fixed(st, y, h * frac)
                    xev = SVector(yev[1], yev[2], yev[3], yev[4])
                    pev = SVector(yev[5], yev[6], yev[7], yev[8])
                    plev = _mcb_plasma_at(st, xev, fl, gp)
                    if plev.inside
                        p_new, rng = _mc_scatter_mom(pev, plev.ucon, plev.ucov, plev.bcon,
                                                     plev.theta_e, plev.gcov, plev.gcon, rng)
                        nspawn += 1
                        slot = (KernelAbstractions.@atomic out.ctr[1] += Int64(1)) - Int64(1)
                        if slot < q.cap
                            k = Int(slot) + 1
                            @inbounds begin
                                out.x[k] = xev
                                out.p[k] = p_new
                                out.w[k] = weight_scat * exp(-dtau_a * frac)
                                out.tau[k] = SVector(tau_a, tau_s)
                                out.n[k] = nscatt + Int32(1)
                                out.rng[k] = _MCM._mc_child_rng(rng, nspawn)
                            end
                        end
                    end
                end
            end
            ns0 = ns1
        end
        w *= exp(-(dtau_a + dtau_s))
        tau_a += dtau_a
        tau_s += dtau_s
        y = y2; x = xn; p = pn; na0 = na1
        if q.scattering && tau_a > q.tau_abs_max                # absorbed: drop
            status = 5
            break
        end
    end
    if status == 2
        KernelAbstractions.@atomic acc.esc[1] += w * (-p[1])
        KernelAbstractions.@atomic acc.nrec[1] += Int64(1)
        _mcb_record!(acc, st, x, p, w, q)
    elseif status == 4
        KernelAbstractions.@atomic acc.ninvalid[1] += Int64(1)
    elseif status == 3
        KernelAbstractions.@atomic acc.ncap_steps[1] += Int64(1)
        KernelAbstractions.@atomic acc.stall_E[1] += w * (-p[1])
    elseif status == 5
        KernelAbstractions.@atomic acc.nacut[1] += Int64(1)
        KernelAbstractions.@atomic acc.acut_E[1] += w * (-p[1])
    end
    return nothing
end

# Seed packets (order 0): continue each packet's own generation stream, as the CPU loop does.
@kernel function _mcb_seed_kernel!(acc, out, st, fl, gp, logsig, q, @Const(xs), @Const(ps),
                                   @Const(ws), @Const(rngs))
    i = @index(Global)
    @inbounds if i <= q.N
        _mcb_march!(acc, out, st, fl, gp, logsig, q, xs[i], ps[i], ws[i], 0.0, 0.0,
                    Int32(0), rngs[i])
    end
end

# Children of one scattering order, each on the stream its parent derived for it
# (`_mc_child_rng`, as the CPU tracker), so buffer order does not matter.
@kernel function _mcb_layer_kernel!(acc, out, st, fl, gp, logsig, q, @Const(xs), @Const(ps),
                                    @Const(ws), @Const(taus), @Const(ns), @Const(rngs))
    i = @index(Global)
    @inbounds if i <= q.N
        _mcb_march!(acc, out, st, fl, gp, logsig, q, xs[i], ps[i], ws[i], taus[i][1], taus[i][2],
                    ns[i], rngs[i])
    end
end

"""
    mc_spectrum_grmhd_gpu(model::GRMHDModel{<:SuperposedBoostedKerrSchildCartesian}, Ns;
                          t_emit, capture_factor, C, backend=CUDABackend(),
                          capacity_factor=0.5, workgroupsize=64, kwargs...)
        -> GRMHDBinarySpectrumResult

Device version of the binary `mc_spectrum_grmhd`, with the same keywords, physics
(bounded bias, step limit and absorption cut, all
counted) and result. Seed packets are generated on the host exactly as the CPU
driver does and tracked on `backend`; children take the same derived streams, so
every packet draws the same random numbers as on the CPU. On the CPU backend the
results agree up to summation order; on a GPU, device rounding can send a rare
packet near a threshold down a different path. `capacity_factor` sizes each
scattering order's child buffer relative to its parent count; an overflow raises
an error. Load after `using BlackLightPlus` (and `using CUDA` for `CUDABackend()`).
"""
function mc_spectrum_grmhd_gpu(model::GRMHDModel{<:SuperposedBoostedKerrSchildCartesian}, Ns::Real;
                               t_emit::Real, capture_factor::Real, C::Real,
                               backend = CUDABackend(),
                               n_phi::Integer = 1, scattering::Bool = true,
                               seed::Integer = 0x51EED5,
                               incl_deg = (17.0, 50.0, 90.0), half_width_deg::Real = 10.0,
                               gen_nu_min::Real = 1.0e8, gen_nu_max::Real = 1.0e16,
                               spec_nu_min::Real = 1.0e8, spec_nu_max::Real = 1.0e24,
                               n_esamp::Integer = _MCM._MC_N_ESAMP_DEFAULT, n_ebins::Integer = 160,
                               R_record::Real = 1.0e3, max_scatt::Integer = 3, bias0::Real = 1.0,
                               stride::Integer = 1, max_zones::Integer = typemax(Int),
                               r_emit_max::Real = Inf, capacity_factor::Real = 0.5,
                               workgroupsize::Integer = 64,
                               max_steps::Integer = _MCM._GRMHD_MAX_STEPS,
                               bias_K::Real = 1.0, tau_abs_max::Real = 100.0,
                               theta_e2_ref::Union{Nothing,Real} = nothing)
    n_phi >= 1 || throw(ArgumentError("n_phi must be at least 1"))
    max_steps >= 1 || throw(ArgumentError("max_steps must be at least 1"))
    bias_K > 0.0 || throw(ArgumentError("bias_K must be positive"))
    tau_abs_max > 0.0 || throw(ArgumentError("tau_abs_max must be positive"))
    theta_e2_ref === nothing || theta_e2_ref > 0.0 ||
        throw(ArgumentError("theta_e2_ref must be positive"))
    st = model.spacetime
    st.orbit isa BinaryOrbit || throw(ArgumentError("pass a CPU metric containing a BinaryOrbit"))
    cf = Float64(capture_factor)

    # ---- host: zones, tables, counts and seed packets (the CPU recipe) ----
    zones = emission_zones(model; t_emit = t_emit, capture_factor = cf,
                           max_zones = max_zones, stride = stride, r_emit_max = r_emit_max)
    nz = length(zones)
    nz == 0 && error("mc_spectrum_grmhd_gpu: no emitting zones (check cuts / snapshot)")
    band = log(gen_nu_max) - log(gen_nu_min)
    jt = build_zone_emission_table(zones; n_esamp = n_esamp, nu_min = gen_nu_min, nu_max = gen_nu_max)
    wgt = build_weight_table(jt, Float64(Ns) / band)
    Nsz_all, dnmax = zone_counts(jt, wgt)
    ngen = Vector{Int}(undef, nz)
    rr = MCRng(Int(seed), 0)
    for z in 1:nz
        fl0 = floor(Int, Nsz_all[z])
        u, rr = mc_uniform(rr)
        ngen[z] = fl0 + (u < (Nsz_all[z] - fl0) ? 1 : 0)
    end
    offset = vcat(0, cumsum(ngen))
    Ntot = offset[end]
    Ntot > 0 || error("mc_spectrum_grmhd_gpu: no seed packets generated")
    xs = Vector{SVector{4,Float64}}(undef, Ntot); ps = similar(xs)
    ws = Vector{Float64}(undef, Ntot); rngs = Vector{MCRng}(undef, Ntot)
    Threads.@threads for gid in 1:Ntot
        z = searchsortedfirst(offset, gid) - 1
        sp, rng = sample_zone_photon(zones[z], wgt, dnmax[z], MCRng(Int(seed), gid))
        xs[gid] = sp.x; ps[gid] = sp.p; ws[gid] = sp.w; rngs[gid] = rng
    end
    mec2 = ME * CL * CL
    emitted = mec2 * sum(ws[i] * (-ps[i][1]) for i in 1:Ntot)

    Rem = model.R_em
    for z in zones
        Rem = max(Rem, 1.2 * sqrt(z.x[2]^2 + z.x[3]^2 + z.x[4]^2))
    end
    # bias normalisation, exactly as the CPU driver (same summation order):
    # theta_e2_ref if given, else the volume-averaged theta_e^2 of the emitting cells
    if theta_e2_ref === nothing
        sv = 0.0
        st2 = 0.0
        @inbounds for z in 1:nz
            sv += zones[z].dV
            st2 += zones[z].theta_e^2 * zones[z].dV
        end
        te2 = st2 / sv
    else
        te2 = Float64(theta_e2_ref)
    end

    # ---- device data ----
    snap = model.snap
    fl = (blk = _mcb_to(backend, permutedims(snap.data, (5, 1, 2, 3, 4))),
          xlft = _mcb_to(backend, reduce(hcat, snap.x_left)),
          dxb = _mcb_to(backend, reduce(hcat, snap.dx)),
          lo = _mcb_to(backend, reduce(hcat, snap.ext_lo)),
          hi = _mcb_to(backend, reduce(hcat, snap.ext_hi)),
          ni = snap.ni, nj = snap.nj, nk = snap.nk, nmb = snap.nmb)
    orb = st.orbit
    st_d = SuperposedBoostedKerrSchildCartesian(st.q,
               DevBinaryOrbit(orb.q, _mcb_to(backend, orb.times), _mcb_to(backend, orb.data), orb.dt))
    c = model.cfg
    gp = (d_unit = c.d_unit, e_unit = c.e_unit, b_unit = c.b_unit, mu = c.plasma_mu,
          ne_ni = c.plasma_ne_ni, use_p = c.plasma_use_p, rat_low = c.plasma_rat_low,
          rat_high = c.plasma_rat_high, gamma = c.plasma_gamma, gamma_i = c.plasma_gamma_i,
          gamma_e = c.plasma_gamma_e, sigma_cut = c.cut_sigma_max, theta_e_min = c.cut_theta_e_min)
    tbl = scattering ? build_hotcross_table() : HotcrossTable(zeros(Float64, 1, 1))
    logsig = _mcb_to(backend, tbl.logsigma)

    cen = collect(Float64.(incl_deg)); nb = length(cen)
    hw = deg2rad(Float64(half_width_deg))
    lo = [clamp(deg2rad(cen[b]) - hw, 0.0, π) for b in 1:nb]
    hi = [clamp(deg2rad(cen[b]) + hw, 0.0, π) for b in 1:nb]
    nph = Int(n_phi)
    proto = Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max, n_ebins = n_ebins, n_thbins = 1)
    zf(T, dims...) = (A = KernelAbstractions.zeros(backend, T, dims...); A)
    acc = (tot_cnt = zf(Int64, n_ebins), tot_w = zf(Float64, n_ebins), tot_wE = zf(Float64, n_ebins),
           cnt = zf(Int64, nb, nph, n_ebins), sw = zf(Float64, nb, nph, n_ebins),
           swE = zf(Float64, nb, nph, n_ebins), lo = _mcb_to(backend, lo), hi = _mcb_to(backend, hi),
           esc = zf(Float64, 1), nrec = zf(Int64, 1), ninvalid = zf(Int64, 1), ncap_steps = zf(Int64, 1),
           stall_E = zf(Float64, 1), nacut = zf(Int64, 1), acut_E = zf(Float64, 1))
    qbase = (cf = cf, C = Float64(C), R_record = Float64(R_record), Rem = Rem,
             L_unit = model.L_unit, max_steps = Int(max_steps),
             scattering = scattering, max_scatt = Int(max_scatt), bias0 = Float64(bias0),
             theta_e2_mean = te2, bias_K = Float64(bias_K), tau_abs_max = Float64(tau_abs_max),
             lnu_min = proto.lnu_min, dlnu = proto.dlnu, n_ebins = Int(n_ebins),
             n_phi = nph, n_bands = nb)

    mkbuf(n) = (x = KernelAbstractions.allocate(backend, SVector{4,Float64}, max(n, 1)),
                p = KernelAbstractions.allocate(backend, SVector{4,Float64}, max(n, 1)),
                w = KernelAbstractions.allocate(backend, Float64, max(n, 1)),
                tau = KernelAbstractions.allocate(backend, SVector{2,Float64}, max(n, 1)),
                n = KernelAbstractions.allocate(backend, Int32, max(n, 1)),
                rng = KernelAbstractions.allocate(backend, MCRng, max(n, 1)),
                ctr = zf(Int64, 1))

    # ---- order 0: seed packets ----
    cap = scattering ? max(1, ceil(Int, capacity_factor * Ntot)) : 0
    out = mkbuf(cap)
    q = merge(qbase, (N = Ntot, cap = cap))
    _mcb_seed_kernel!(backend, workgroupsize)(acc, out, st_d, fl, gp, logsig, q,
        _mcb_to(backend, xs), _mcb_to(backend, ps), _mcb_to(backend, ws), _mcb_to(backend, rngs);
        ndrange = Ntot)
    KernelAbstractions.synchronize(backend)

    # ---- orders 1..max_scatt: children ----
    layer = 0
    while scattering
        nchild = Int(Array(out.ctr)[1])
        nchild == 0 && break
        nchild > cap && error("mc_spectrum_grmhd_gpu: scattering order $(layer + 1) spawned $nchild children " *
                              "but the buffer holds $cap; raise capacity_factor (dropping them would bias the spectrum)")
        ntrack = nchild
        layer += 1
        parent = out
        cap = max(1, ceil(Int, capacity_factor * ntrack))
        out = mkbuf(cap)
        q = merge(qbase, (N = ntrack, cap = cap))
        _mcb_layer_kernel!(backend, workgroupsize)(acc, out, st_d, fl, gp, logsig, q,
            parent.x, parent.p, parent.w, parent.tau, parent.n, parent.rng; ndrange = ntrack)
        KernelAbstractions.synchronize(backend)
    end
    ninv = Int(Array(acc.ninvalid)[1])
    ninv > 0 && error("mc_spectrum_grmhd_gpu: $ninv packets hit invalid geometry")

    # ---- device bins -> host spectra ----
    cnt = Array(acc.cnt); sw = Array(acc.sw); swE = Array(acc.swE)
    bands = [Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max, n_ebins = n_ebins, n_thbins = 1)
             for _ in 1:nb, _ in 1:nph]
    for b in 1:nb, j in 1:nph
        bands[b, j].count[1, :] .= cnt[b, j, :]
        bands[b, j].sumw[1, :] .= sw[b, j, :]
        bands[b, j].sumwE[1, :] .= swE[b, j, :]
    end
    total = Spectrum(; nu_min = spec_nu_min, nu_max = spec_nu_max, n_ebins = n_ebins, n_thbins = 1)
    total.count[1, :] .= Array(acc.tot_cnt); total.sumw[1, :] .= Array(acc.tot_w)
    total.sumwE[1, :] .= Array(acc.tot_wE)
    dOmega = [2.0 * π * (cos(lo[b]) - cos(hi[b])) / nph for b in 1:nb, _ in 1:nph]
    phi_deg = [360.0 * (j - 1) / nph for j in 1:nph]
    return _MCM.GRMHDBinarySpectrumResult(cen, Float64(half_width_deg), phi_deg, dOmega, bands, total,
                                     emitted, mec2 * Array(acc.esc)[1], nz, Ntot,
                                     Int(Array(acc.nrec)[1]), scattering, Float64(t_emit),
                                     Int(max_steps), Int(Array(acc.ncap_steps)[1]),
                                     mec2 * Array(acc.stall_E)[1],
                                     te2, Float64(bias_K), Float64(tau_abs_max),
                                     Int(Array(acc.nacut)[1]), mec2 * Array(acc.acut_E)[1])
end
