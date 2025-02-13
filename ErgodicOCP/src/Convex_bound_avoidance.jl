module ConvexBoundAvoidance

## Kaleb Ben Naveed August 2024

using Plots, StaticArrays, LinearAlgebra, Statistics
using LazySets

export ConvexPolygon

struct ConvexPolygon
    polygon::VPolygon{Float64, Vector{Float64}}
    vertices::Matrix{Float64}
    edges::Matrix{Float64}
end

function ConvexPolygon(VP_Polygon, vertices::Matrix{Float64})
    # Get the number of the vertices 
    n = size(vertices, 2)
    # Set edges wit the last one connected to the first one
    edges = [vertices[:, i] - vertices[:, mod(i, n) + 1] for i in 1:n]
    return ConvexPolygon(VP_Polygon, vertices, hcat(edges...))
end


# Function to compute the distance from a point to a line segment and return the closest point on the line segment
function point_to_line_distance(p, v1, v2)
    v_vec = v2 - v1
    p_vec = p - v1
    v_vec_len = norm(v_vec)
    v_unitvec = v_vec / v_vec_len
    projection_length = dot(p_vec, v_unitvec)
    if projection_length < 0
        return norm(p - v1), v1
    elseif projection_length > v_vec_len
        return norm(p - v2), v2
    else
        projection_p = v1 + projection_length * v_unitvec
        return norm(p - projection_p), projection_p
    end
end


# Function to compute the minimum distance from a point to the boundary of a convex polygon
function minimum_distance_to_boundary(polygon, p)
    n = size(polygon.vertices, 2)
    min_distance = Inf
    closest_point = nothing
    for i in 1:n
        v1 = polygon.vertices[:, i]
        v2 = polygon.vertices[:, mod(i, n) + 1]
        distance, point_on_segment = point_to_line_distance(p, v1, v2)
        if distance < min_distance
            min_distance = distance
            closest_point = point_on_segment
        end
    end
    return min_distance, closest_point
end

# For debugging purposes. 
function plot_polygon_and_point(polygon, point::Vector{Float64}, closest_point::Vector{Float64}, centroid::Vector{Float64})
    polygon_vertices = hcat(polygon.vertices, polygon.vertices[:, 1])  # Close the polygon
    plot(polygon_vertices[1, :], polygon_vertices[2, :], seriestype=:shape, fillalpha=0.2, label="Convex Polygon")
    scatter!([point[1]], [point[2]], color=:red, label="Point")
    scatter!([closest_point[1]], [closest_point[2]], color=:blue, label="Closest Point")
    scatter!([centroid[1]], [centroid[2]], color=:green, label="Centroid")
    xlabel!("x")
    ylabel!("y")
    title!("Convex Polygon and Point")
#     legend(:topright)
end


# Function to calculate the centroid of the polygon
function calculate_centroid(polygon)
    vertices = polygon.vertices
    centroid = mean(vertices, dims=2)
    return centroid[:]
end


# Function to compute the normal vector pointing from the closest point to the centroid
function normal_vector_to_centroid(polygon, closest_point::Vector{Float64})
    centroid = calculate_centroid(polygon)
    normal_vector = centroid - closest_point
    return normal_vector / norm(normal_vector)  # Normalize the vector
end



end