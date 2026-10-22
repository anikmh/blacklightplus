# Geodesics — binary_orbits.jl

"""
    BinaryOrbit

Tabulated binary-BH orbital data in units G = c = M1 = 1,
with mass ratio `q = M2/M1`.

Load from a file using `BinaryOrbit(filename; t0=0.0)`.
The `data` array has shape (18, nt), with rows ordered as:

    1:3    BH 1 position
    4:6    BH 2 position
    7:9    BH 1 spin
    10:12  BH 2 spin
    13:15  BH 1 velocity
    16:18  BH 2 velocity

Positions, velocities, and spins are spatial three-vectors.
Stored spins are `a = J/M` in primary-mass units.

Call `orbit(t)` to interpolate both BHs' orbital quantities at
simulation coordinate time `t`.
"""
struct BinaryOrbit
    q::Float64
    times::Vector{Float64}
    data::Matrix{Float64}       # (18, nt)
    dt::Float64
end

"""
    BinaryOrbit(filename::AbstractString; t0=0.0)

Load an Athena++ orbit file containing a text header followed by
little-endian `Float64` records.

Add `t0` to the tabulated times and scale the secondary spin by `q`
to express `a = J/M` in primary-mass units.
Require at least two samples and positive initial time spacing;
interpolation assumes uniformly spaced records.
"""
function BinaryOrbit(filename::AbstractString; t0::Real=0.0)
    q, records = open(filename, "r") do io
        header = split(readline(io))
        nt = parse(Int, header[1])
        q = parse(Float64, header[2])

        nt >= 2 || error("orbit file requires at least two time samples")

        # Read binary bit patterns, then convert from little-endian.
        bits = Matrix{UInt64}(undef, 19, nt)
        read!(io, bits)

        eof(io) || error("unexpected trailing data in orbit file")

        records = reinterpret(Float64, ltoh.(bits))
        return q, records
    end

    times = records[1, :] .+ Float64(t0)
    data = records[2:19, :]

    # Convert secondary spin to a = J/M in units of M1.
    # Apply this exactly once, when loading.
    data[10:12, :] .*= q

    dt = times[2] - times[1]
    dt > 0.0 || error("orbit time spacing must be positive")

    return BinaryOrbit(q, times, data, dt)
end

"""
    (orbit::BinaryOrbit)(t::Real)

Interpolate orbital data at simulation coordinate time `t`.

Return `(center=(c1, c2), velocity=(v1, v2), spin=(a1, a2))`.
Each entry is a spatial three-vector of type `SVector{3,Float64}`;
spins are `a = J/M` in primary-mass units.

Use Athena++'s uniform-spacing interval selection and linear
interpolation. Outside the tabulated range, extrapolate using
the first or last segment.

Interpolate velocities from stored values rather than differentiating
the positions.
"""
function (orbit::BinaryOrbit)(t::Real)
    time = Float64(t)
    nt = length(orbit.times)

    # Reproduce Athena++'s zero-based interval calculation.
    i0 = trunc(Int, (time - orbit.times[1]) / orbit.dt + 1000) - 1000
    i0 = clamp(i0, 0, nt - 1)

    # Convert to Julia's one-based indexing.
    i = i0 + 1

    left = i == nt ? nt - 1 : i
    right = i == nt ? nt : i + 1

    # Athena++ anchors the extrapolation at the last sample
    # when i == nt, and at the left sample otherwise.
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
    _freeze_orbit(orbit::BinaryOrbit, t) -> BinaryOrbit

Return `orbit` held at its state at simulation coordinate time `t`.

The result stores that state twice, so calling it at any time returns
the positions, velocities, and spins of `orbit(t)`.
"""
function _freeze_orbit(orbit::BinaryOrbit, t::Real)
    s = orbit(t)
    col = Float64[s.center[1]; s.center[2]; s.spin[1]; s.spin[2];
                  s.velocity[1]; s.velocity[2]]
    time = Float64(t)
    return BinaryOrbit(orbit.q, [time - 1.0, time + 1.0], hcat(col, col), 2.0)
end
