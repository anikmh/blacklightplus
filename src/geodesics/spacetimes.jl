# Geodesics — spacetimes.jl
#
# GENERAL analytic spacetimes beyond single Kerr. The payoff of the AD design
# (contrast hand-differentiated per-metric providers as in blacklight): a new
# spacetime only defines `metric(st, x)`; the geodesic engine's `inverse_metric`
# (StaticArrays `inv`) and `inverse_metric_gradient` (ForwardDiff through
# `inverse_metric`) then work UNCHANGED, and `geodesic_rhs`, `trace_geodesic`,
# `render`, the GPU tracer, and AD all compose automatically.

using StaticArrays

# ---------------------------------------------------------------- superposed KS binary
"""
    SuperposedKerrSchild(masses, spins, centers)

Analytic superposed Kerr-Schild binary: two (or one) static Kerr-Schild black
holes with spins along ±z,

    g_μν = η_μν + Σ_A f_A l^A_μ l^A_ν      (covariant l^A_0 = 1),

exact for one BH, the standard large-separation binary initial-data ansatz for
two. The single-KS inverse shortcut is invalid for the sum (cross term
f_A f_B l^A·l^B), so `inverse_metric` uses the full 4x4 inverse (StaticArrays)
and `inverse_metric_gradient` differentiates THAT with ForwardDiff — reducing
exactly to single Kerr when a second mass is zero.
"""
struct SuperposedKerrSchild{N,T} <: StationarySpacetime{T}
    mass::SVector{N,T}
    spin::SVector{N,T}
    center::SVector{N,SVector{3,T}}
end

function SuperposedKerrSchild(masses::AbstractVector, spins::AbstractVector,
                              centers::AbstractVector)
    n = length(masses)
    @assert length(spins) == n == length(centers)
    ctrs = SVector{n}(SVector{3,Float64}(c...) for c in centers)
    return SuperposedKerrSchild(SVector{n,Float64}(masses...),
                                SVector{n,Float64}(spins...), ctrs)
end

# One Kerr-Schild hole's (r, f, l_a) comes from the shared `_ks_term` primitive
# (metrics.jl) — the same closed form single Kerr uses, evaluated at the per-hole
# offset. Superposition then sums f_A l^A_μ l^A_ν over the holes.
@inline function metric(st::SuperposedKerrSchild{N}, x::SVector{4,T}) where {N,T}
    g = ETA
    @inbounds for A in 1:N
        c = st.center[A]
        _, f, l = _ks_term(x[2] - c[1], x[3] - c[2], x[4] - c[3], st.mass[A], st.spin[A])
        g = g + f * (l * l')
    end
    return g
end

@inline inverse_metric(st::SuperposedKerrSchild, x::SVector{4}) = inv(metric(st, x))

"Local KS radius of hole A (for termination / diagnostics)."
@inline function ks_radius_hole(st::SuperposedKerrSchild, x::SVector{4}, A::Int)
    c = st.center[A]
    return _ks_term(x[2] - c[1], x[3] - c[2], x[4] - c[3], st.mass[A], st.spin[A])[1]
end

"Per-hole horizon radius M_A + sqrt(M_A^2 - a_A^2)."
@inline function horizon_radius_hole(st::SuperposedKerrSchild, A::Int)
    m, a = st.mass[A], st.spin[A]
    d = m * m - a * a
    return m + (d > 0 ? sqrt(d) : zero(m))
end

# trace_geodesic for the binary: terminate when any hole is entered, or on escape.
function trace_geodesic(st::SuperposedKerrSchild{N}, y0::SVector{8}; λmax=1.0e5,
                        reltol=1e-10, abstol=1e-12, escape=ESCAPE_RADIUS,
                        horizon_eps=HORIZON_EPS_SAMPLED,
                        solver=Vern7(), save_everystep=false) where {N}
    rterm = SVector{N}(horizon_radius_hole(st, A) + horizon_eps for A in 1:N)
    prob = ODEProblem(_rhs_ode, y0, (0.0, λmax), st)
    function captured(u, λ, i)
        x = SVector(u[1], u[2], u[3], u[4])
        m = Inf
        @inbounds for A in 1:N
            m = min(m, ks_radius_hole(st, x, A) - rterm[A])
        end
        return m
    end
    esc(u, λ, i) = sqrt(u[2]^2 + u[3]^2 + u[4]^2) - escape
    cb = CallbackSet(ContinuousCallback(captured, terminate!),
                     # escape only on the OUTBOUND upcrossing (see integrator.jl)
                     ContinuousCallback(esc, terminate!, nothing))
    return solve(prob, solver; reltol, abstol, callback=cb,
                 save_everystep, save_start=save_everystep)
end

"""
    trace_rays_sampled(st::SuperposedKerrSchild, rays; C=40.0, escape=ESCAPE_RADIUS,
                       nmax=6000, R_em=Inf, max_step=Inf, cell_size_fn=nothing,
                       cell_frac=1.0) -> (states, dls)

Binary sibling of the Kerr fixed-control sampled tracer (integrator.jl): the
same step rule and source→camera output contract, generalized to N punctures.
Two changes vs the Kerr method — (i) the step uses the distance to the NEAREST
horizon, `h = min_A(r_A - r_h_A)/C`, so sampling densifies toward whichever hole
a ray approaches; (ii) capture (at horizon + `HORIZON_EPS_SAMPLED`) terminates
on EITHER hole, matching `trace_geodesic(::SuperposedKerrSchild)`. Escape uses
the global coordinate radius. `R_em` and the opt-in `max_step`/`cell_size_fn`
AMR caps behave exactly as in the Kerr method, and its step-density accuracy
NOTE applies unchanged.
"""
function trace_rays_sampled(st::SuperposedKerrSchild{N}, rays::Vector{SVector{8,Float64}};
                            C::Float64=40.0, escape::Float64=ESCAPE_RADIUS, nmax::Int=6000,
                            R_em::Float64=Inf, max_step::Float64=Inf,
                            cell_size_fn=nothing, cell_frac::Float64=1.0) where {N}
    n = length(rays)
    rh = SVector{N}(horizon_radius_hole(st, A) for A in 1:N)
    states = Vector{Vector{SVector{8,Float64}}}(undef, n)
    dls = Vector{Vector{Float64}}(undef, n)
    Threads.@threads for i in 1:n
        ys = SVector{8,Float64}[rays[i]]
        hs = Float64[]
        sizehint!(ys, 1024)                       # see the Kerr method's sizehint! NOTE
        sizehint!(hs, 1024)
        y = rays[i]
        r_prev = Inf
        for _ in 1:nmax
            x = SVector(y[1], y[2], y[3], y[4])
            # distance to the nearest HORIZON; capture (either hole) at horizon + eps,
            # exactly the Kerr split — stepping by horizon distance while terminating
            # eps outside it keeps h >= eps/C, so capture lands in finitely many steps
            dmin = Inf
            @inbounds for A in 1:N
                dA = ks_radius_hole(st, x, A) - rh[A]
                isfinite(dA) || (dmin = -Inf; break)
                dmin = min(dmin, dA)
            end
            dmin < HORIZON_EPS_SAMPLED && break
            R = sqrt(x[2]^2 + x[3]^2 + x[4]^2)
            ((R > escape || R > R_em) && R > r_prev) && break   # outbound escape only
            r_prev = R
            h = min(dmin / C, max_step)
            if cell_size_fn !== nothing
                h = min(h, cell_frac * cell_size_fn(x))
            end
            y = rk4_step_fixed(st, y, h)
            push!(ys, y)
            push!(hs, h)
        end
        m = length(ys)
        rev = Vector{SVector{8,Float64}}(undef, m)
        dl = Vector{Float64}(undef, m)
        for k in 1:m
            rev[k] = ys[m+1-k]                    # integration order → source→camera
        end
        for k in 1:m-1
            dl[k] = hs[m-k]
        end
        dl[m] = 0.0
        states[i] = rev
        dls[i] = dl
    end
    return states, dls
end


"""
    SuperposedBoostedKerrSchildCartesian(q, orbit)

Time-dependent superposed boosted Kerr-Schild binary metric in Cartesian
coordinates, following the Athena++ binary-metric implementation.

Uses coordinates x = (t, x, y, z), signature (-,+,+,+), and units
G = c = M₁ = 1. The BH masses are M₁ = 1 and M₂ = q.

The callable `orbit(t)` must return a named tuple:

    (
        center   = (c1, c2),
        velocity = (v1, v2),
        spin     = (a1, a2),
    )

Each cA, vA, and aA is an `SVector{3,Float64}`. Positions, velocities,
and spins are evaluated at simulation coordinate time t.

Spin vectors are a = J/M, where a and J are three-vectors.
The vector a has dimensions of length; χ = a/M is dimensionless.
Any secondary-spin mass scaling must already be applied by `orbit(t)`.

The metric is assembled as

    g_{μν} = η_{μν} + f₁ ℓ⁽¹⁾_μ ℓ⁽¹⁾_ν + f₂ ℓ⁽²⁾_μ ℓ⁽²⁾_ν.

Includes Athena++'s small-position adjustment, interior radius
regularization, and finite-difference derivative prescription.
"""
struct SuperposedBoostedKerrSchildCartesian{F} <: AnalyticSpacetime{Float64}
    q::Float64
    orbit::F
end

SuperposedBoostedKerrSchildCartesian(q::Real, orbit) =
    SuperposedBoostedKerrSchildCartesian{typeof(orbit)}(
        Float64(q), orbit
    )

# Numerical prescriptions used in the Athena++ implementation.
const _SBKS_SMALL = 1.0e-7
const _SBKS_DELTA = 1.0e-4


# 1. Reproduce Athena++'s small-a⋅X position adjustment.
#
# The returned D is deliberately assigned ±SMALL in this branch.
# It is not necessarily exactly dot(a, X) after the adjustment.
function _sbks_adjust_position(X, a)
    amag = sqrt(dot(a, a))
    D = dot(a, X)

    if abs(D) < _SBKS_SMALL
        target = D >= 0.0 ? _SBKS_SMALL : -_SBKS_SMALL
        shift = target - D / (amag + _SBKS_SMALL)

        X = X + shift * a / (amag + _SBKS_SMALL)
        D = target
    end

    return X, D
end


# 2. Transform the displacement into the instantaneous BH rest frame.
function _sbks_rest_position(position, center, velocity, spin)
    d = position - center

    v2 = dot(velocity, velocity)
    lorentz = inv(sqrt(1.0 - v2))
    K = lorentz^2 / (lorentz + 1.0)

    X = d + K * dot(velocity, d) * velocity

    # Matches the adjustment in Athena++'s get_prime_coords.
    X, _ = _sbks_adjust_position(X, spin)

    return X, lorentz, K
end


# 3. Calculate the KS radius and auxiliary angles used by the
#    interior regularization.
function _sbks_radius_angles(X, a)
    amag = sqrt(dot(a, a))

    # Athena++ calculates this cross product before adjusting
    # the local coordinate copy. Preserve that ordering.
    a_cross_X = cross(a, X)

    X_adjusted, D = _sbks_adjust_position(X, a)

    R = sqrt(dot(X_adjusted, X_adjusted))
    difference = R^2 - amag^2

    r = sqrt(
        difference + sqrt(difference^2 + 4.0 * D^2)
    ) / sqrt(2.0)

    lspace = (
        r * X_adjusted - a_cross_X + (D / r) * a
    ) / (r^2 + amag^2)

    # These angles describe lspace's direction in the fixed axes.
    # For spin aligned with +z, they give the usual KS angles.
    theta_aux = acos(clamp(lspace[3], -1.0, 1.0))
    phi_aux = atan(lspace[2], lspace[1])

    return r, theta_aux, phi_aux
end


# 4. Calculate one BH's covariant metric contribution f * l_mu * l_nu.
function _sbks_bh_contribution(mass, center, velocity, spin, position)
    X, lorentz, K =
        _sbks_rest_position(position, center, velocity, spin)

    # Matches the additional adjustment inside Athena++'s
    # boosted_BH_metric_addition.
    X, _ = _sbks_adjust_position(X, spin)

    r, theta_aux, phi_aux = _sbks_radius_angles(X, spin)

    a2 = dot(spin, spin)
    r_horizon = mass + sqrt(mass^2 - a2)

    # Reproduce Athena++'s interior metric prescription.
    if r < 0.8 * r_horizon
        r = 0.8 * r_horizon

        n = SVector(
            sin(theta_aux) * cos(phi_aux),
            sin(theta_aux) * sin(phi_aux),
            cos(theta_aux),
        )

        X = r * n + cross(spin, n)
    end

    D = dot(spin, X)

    # Kerr–Schild scalar coefficient.
    f = 2.0 * mass * r^3 / (r^4 + D^2)

    # Spatial components of the rest-frame covector.
    # Its temporal component is lprime_0 = 1.
    lprime = (
        r * X - cross(spin, X) + (D / r) * spin
    ) / (r^2 + a2)

    # Transform the covector into simulation coordinates.
    velocity_dot_l = dot(velocity, lprime)

    l0 = lorentz * (1.0 - velocity_dot_l)
    lspace = lprime + (K * velocity_dot_l - lorentz) * velocity

    # Four covariant components, ordered (t, x, y, z).
    l = SVector(l0, lspace[1], lspace[2], lspace[3])

    return f * (l * transpose(l))
end


# 5. Assemble the time-dependent binary metric.
function metric(
    st::SuperposedBoostedKerrSchildCartesian,
    x::SVector{4},
)
    t = x[1]
    position = SVector(x[2], x[3], x[4])

    orb = st.orbit(t)

    h1 = _sbks_bh_contribution(
        1.0,
        orb.center[1],
        orb.velocity[1],
        orb.spin[1],
        position,
    )

    h2 = _sbks_bh_contribution(
        st.q,
        orb.center[2],
        orb.velocity[2],
        orb.spin[2],
        position,
    )

    return ETA + h1 + h2
end


# 6. Invert the combined metric.
function inverse_metric(
    st::SuperposedBoostedKerrSchildCartesian,
    x::SVector{4},
)
    return inv(metric(st, x))
end


# 7. Covariant metric derivatives.
#
# Return convention:
#   dg[mu, nu, sigma] = ∂g_{mu nu} / ∂x^sigma
#
# Julia indices 1,2,3,4 correspond to coordinates t,x,y,z.
# Uses Athena++'s forward-difference increment and interior cutoff.
function metric_gradient(
    st::SuperposedBoostedKerrSchildCartesian,
    x::SVector{4,Float64},
)
    position = SVector(x[2], x[3], x[4])
    orb = st.orbit(x[1])

    dg = zeros(MArray{Tuple{4,4,4},Float64,3,64})

    # Athena++ sets every metric derivative to zero in this region.
    for A in 1:2
        X, _, _ = _sbks_rest_position(
            position,
            orb.center[A],
            orb.velocity[A],
            orb.spin[A],
        )

        R = sqrt(dot(X, X))
        amag = sqrt(dot(orb.spin[A], orb.spin[A]))

        if R <= amag
            return SArray{Tuple{4,4,4},Float64,3,64}(dg)
        end
    end

    g = metric(st, x)

    for sigma in 1:4
        shifted = setindex(
            x,
            x[sigma] + _SBKS_DELTA,
            sigma,
        )

        # Use the actual floating-point increment, as Athena++ does.
        step = shifted[sigma] - x[sigma]

        gp = metric(st, shifted)

        for nu in 1:4, mu in 1:4
            dg[mu, nu, sigma] = (gp[mu, nu] - g[mu, nu]) / step
        end
    end

    return SArray{Tuple{4,4,4},Float64,3,64}(dg)
end


# 8. Inverse-metric derivatives required by the Hamiltonian integrator.
#
# ∂g⁻¹ = -g⁻¹ (∂g) g⁻¹
function inverse_metric_gradient(
    st::SuperposedBoostedKerrSchildCartesian,
    x::SVector{4,Float64},
)
    ginv = inverse_metric(st, x)
    dg = metric_gradient(st, x)

    dginv = zeros(MArray{Tuple{4,4,4},Float64,3,64})

    for sigma in 1:4
        derivative = SMatrix{4,4,Float64,16}(
            dg[:, :, sigma]
        )

        inverse_derivative = -ginv * derivative * ginv

        for nu in 1:4, mu in 1:4
            dginv[mu, nu, sigma] = inverse_derivative[mu, nu]
        end
    end

    return SArray{Tuple{4,4,4},Float64,3,64}(dginv)
end