module TrajectoryLibrary

using StaticArrays

# Function to generate smooth transitions between rows
function smooth_transition(x1, y1, x2, y2, steps)
    t = range(0, stop=1, length=steps)
    x = (1 .- t) .* x1 .+ t .* x2
    y = (1 .- t) .* y1 .+ t .* y2
    return x, y
end

# Generate the smooth lawnmower path
function generate_smooth_lawnmower_path(length, width, mowing_width, top_speed,sampling_interval, total_points)

    distance_per_sample = top_speed * sampling_interval

    x_coords = Float64[]
    y_coords = Float64[]
    steps_per_row = Int(round(length / distance_per_sample))
    
    # Initialize starting coordinates
    push!(x_coords, 0.0)
    push!(y_coords, 0.0)
    
    for i in 0:mowing_width:width
        if size(x_coords, 1) >= total_points - 1
            break
        end
        if mod(floor(i / mowing_width), 2) == 0
            # Mow from left to right
            new_x_coords = collect(range(0, stop=length, length=steps_per_row + 1))
            new_y_coords = fill(i, steps_per_row + 1)
        else
            # Mow from right to left
            new_x_coords = collect(range(length, stop=0, length=steps_per_row + 1))
            new_y_coords = fill(i, steps_per_row + 1)
        end
        
        append!(x_coords, new_x_coords[2:end])  # Skip the first element to avoid duplication
        append!(y_coords, new_y_coords[2:end])  # Skip the first element to avoid duplication
        if size(x_coords, 1) >= total_points - 1
            break
        end
        if i + mowing_width <= width
            x_smooth, y_smooth = smooth_transition(new_x_coords[end], new_y_coords[end], new_x_coords[end], i + mowing_width, steps_per_row)
            append!(x_coords, x_smooth)
            append!(y_coords, y_smooth)
        end
    end
    
    x0 = [@SVector[x_coords[1], y_coords[1]]]
    lawn_traj = [x0,]
    for i in 2:size(x_coords, 1)
        push!(lawn_traj, [@SVector[x_coords[i], y_coords[i]]])
    end


    return lawn_traj
end


end