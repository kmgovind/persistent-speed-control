module Transects

using LinearAlgebra
import TravelingSalesmanHeuristics
include("jordan_lake_domain.jl")

function create_points(grid_points)
    bounded_pts = Vector{Vector{Float64}}()
    for point in grid_points
        if point ∈ JordanLakeDomain.convex_polygon.polygon
            push!(bounded_pts, point)
        end
    end

    new_jordan_pts = solve_basic(bounded_pts)
    return new_jordan_pts
end


# pts is a vector of waypoints to go to
# returns the order of visiting these points
# bias > 0 means the cost of going along x is slightly less than the cost of going along y
function solve_tsp(pts::Vector{P}; bias=0.05) where {F, P<: AbstractVector{F}}
    
    N = length(pts)
    D = zeros(F, N, N)
    
    for i=1:N, j=(i+1):N
        D[i, j] = norm(pts[i] - pts[j]) + bias * abs(pts[i][2] - pts[j][2])
        D[j, i] = D[i, j]
    end
    
    inds = TravelingSalesmanHeuristics.solve_tsp(D)[1]
    
    # return the sorted points
    return pts[inds]
    
end

struct PointSet{F}
    xs::Vector{F}
    y::F
end

function solve_basic(pts::Vector{P}) where {F, P<:AbstractVector{F}}
    
    N = length(pts)
    
    # group by unique ys
    unique_ys = last.(pts) |> unique |> sort
    pt_sets = [PointSet(F[], y) for y in unique_ys]
    
    for pt in pts
        ind = searchsortedfirst(unique_ys, pt[2])
        push!(pt_sets[ind].xs, pt[1])
    end
    
    # sort the pointsets, alternating the direction of sorting
    for (i, s) in enumerate(pt_sets)
        if mod(i, 2) == 0
            sort!(s.xs)
        else
            sort!(s.xs, rev=true)
        end
    end
    
    # now join all the points
    path = P[]
    for s in pt_sets
        for x in s.xs
            push!(path, [x, s.y])
        end
    end
    
    return path
    
end

end
