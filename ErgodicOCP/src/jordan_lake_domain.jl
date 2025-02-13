module JordanLakeDomain
export 𝐬 # Vector of vectors of x- and y-coordinates in units of km
export convex_polygon # Boundary of region of interest in Jordan lake
export grid_mesh, meshx, meshy

using LazySets
include("Convex_bound_avoidance.jl")

const DIST_BT_LOCS = 0.15 # Horizontal/vertical distance between locations [km]
# Note: `DIST_BT_LOCS = 0.2` produces a `length(𝐬)` of 114
#       `DIST_BT_LOCS = 0.15` produces a `length(𝐬)` of 203
#       `DIST_BT_LOCS = 0.1` produces a `length(𝐬)` of 458

# const polygon = VPolygon([[0, 0],
#     [0.3362, 0],
#     [0.7925, 4.3707],
#     [0.9366, 6.3399],
#     [-0.1441, 4.7069],
#     [-0.3602, 1.3929],
# ])

# VP_Polygon = VPolygon([[0.4, 0],
#     [0.7362, 0],
#     [1.1925, 4.3707],
#     [1.3366, 6.3399],
#     [0.2559, 4.7069],
#     [0.0398, 1.3929],
# ])

# vertices = [0.4 0.7362 1.1925 1.3366 0.2559 0.0398;
#             0 0 4.3707 6.3399 4.7069 1.3929]

VP_Polygon = VPolygon([[0.01, 0.01],
    [0.9723, -0.061],
    [1.4687, 1.7472],
    [0.078, 1.5507],
])

vertices = [0.01 0.9723 1.4687 0.078;
            0.01 -0.061 1.7472 1.5507]


# Create a ConvexPolygon object
convex_polygon = ConvexBoundAvoidance.ConvexPolygon(VP_Polygon, vertices)
grid_mesh = Vector{Vector{Float64}}() # Domain Grid Points
meshx = Vector{Float64}()
meshy = Vector{Float64}()

𝐬 = Vector{Vector{Float64}}() # points of polygon
M = length(𝐬)

# function populate_domain!(xstart, xstop, ystart, ystop, sdim)
#     xvals = range(xstart, stop=xstop, length=sdim)
#     yvals = range(ystart, stop=ystop, length=sdim)
#     points = [[x, y] for y in yvals for x in xvals]
#     for p in points
#         push!(𝐬, p)
#     end
#     M = length(𝐬)
# end

let
    bounding_box = overapproximate(VP_Polygon)
    c = bounding_box.center
    r = bounding_box.radius

    mesh_x = range(c[1] - r[1], c[1] + r[1], step=DIST_BT_LOCS)
    mesh_y = range(c[2] - r[2], c[2] + r[2], step=DIST_BT_LOCS)

    points_of_bounding_box = [[x, y] for y in mesh_y for x in mesh_x]

    for p in points_of_bounding_box
        if p ∈ VP_Polygon
            push!(𝐬, p)
        end
        push!(grid_mesh, p)
    end

    for p in mesh_x
        push!(meshx, p)
    end

    for p in mesh_y
        push!(meshy, p)
    end

    # sdim = 100; # number of data points along each axis in the domain (defining a square domain)
    # xstart, xstop = 0, 1.4; # x-axis start and end positions
    # ystart, ystop = 0, 6.5; # y-axis start and end positions

    # xvals = range(xstart, stop=xstop, length=sdim)
    # yvals = range(ystart, stop=ystop, length=sdim)

    # points = [[x,y] for y in yvals for x in xvals]

    # for p in points
    #     push!(𝐬, p)
    # end
end

end
