# Geodesics — binary_orbits.jl

"""
    BinaryOrbit(filename; t0=0.0)

Read an Athena++ binary-BH orbit file.

The first line contains the number of samples and mass ratio q.
Each subsequent binary record contains 19 little-endian Float64 values:
time, followed by 18 orbital quantities.

The orbital quantities are stored in this order:

    1:3    BH 1 position
    4:6    BH 2 position
    7:9    BH 1 spin
    10:12  BH 2 spin
    13:15  BH 1 velocity
    16:18  BH 2 velocity

Adds t0 to the stored times and multiplies BH 2's spin components
by q, matching Athena++'s set_orbit_arrays.

The resulting object is callable: orbit(t) returns interpolated
positions, velocities, and spin vectors for both BHs.
"""
struct BinaryOrbit
    q::Float64
    times::Vector{Float64}
    data::Matrix{Float64}       # (18, nt)
    dt::Float64
end

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

Interpolate the orbital quantities at simulation coordinate time t.

Returns

    (
        center   = (c1, c2),
        velocity = (v1, v2),
        spin     = (a1, a2),
    )

Each cA, vA, and aA is an SVector{3,Float64}.

Uses Athena++'s uniform-spacing interval selection and componentwise
linear interpolation. Outside the tabulated time range, extrapolates
using the first or last segment, as Athena++ does.

Velocities are interpolated from their stored values, rather than
calculated by differentiating the positions.
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