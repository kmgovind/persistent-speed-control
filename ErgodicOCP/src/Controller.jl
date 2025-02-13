module Controller

struct BoatState
    position_x::Float64
    position_y::Float64
    heading::Float64
    state_of_charge::Float64
end

struct Waypoint
    times::Vector{Float64}
    positions_x::Vector{Float64}
    positions_y::Vector{Float64}
end

struct ControlInput
    forward_velocity::Float64 # [m/s]
    heading_angle::Float64 # [rad] - relative to +x axis, CCW
end

struct InformationEstimate
    # grid::Grid
    wind_x::Matrix
    wind_y::Matrix
end

end # module Controller
