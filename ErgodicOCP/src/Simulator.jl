module Simulator

using Revise

# expose other modules to package
include("Controller.jl")
include("Utilities.jl")
include("SyntheticData.jl")

using .Controller
using .Utilities
using .SyntheticData

# TODO: Define grid structure
# struct Grid
#     origin
#     dxs
#     Ls
# end

struct Measurement
    measurement_time::Float64
    wind_x::Float64
    wind_y::Float64
end

# TODO: Define MissionDomain structure
# struct MissionDomain
# end

# TODO: Complete step method
# function step(current_time, current_boat_state, control_input)
# end

# TODO: Complete measure method
# function measure(time, position)
# end

end # module Simulator