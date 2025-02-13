module ErgodicController

using FFTW, LinearAlgebra, StaticArrays
using RecipesBase

include("Convex_bound_avoidance.jl")

using .ConvexBoundAvoidance


struct Grid{T, F}
    o::T # origin of the grid
    dx::T # spacing of each cell 
    N::Tuple{Int, Int} # total extent of the simulation domain
    dct_plan::F # Creates a plan for FFTW 
end

function Grid(o::T, dx::T, L::T) where {T}
    Ns = ntuple(i->Int(ceil(1 + L[i] / (dx[i]))), 2)
    return Grid(o, dx, Ns)
end

function Grid(o, dx, N)
        M = zeros(N)
        dct_plan = FFTW.plan_r2r(M, FFTW.REDFT10)

    os = @SVector [o[1], o[2]]
    dxs = @SVector [dx[1], dx[2]]
    Grid(os, dxs, N, dct_plan)
end

function lengths(grid::G) where {G <: Grid}
    return @SVector [ (grid.N[i]-1) * grid.dx[i] for i=1:2]
end

function center(grid::G) where {G <: Grid}
    return grid.o + 0.5 * lengths(grid)
end

function xs(grid::G) where {G <: Grid}
    return grid.o[1] .+ (0:(grid.N[1]-1)) * grid.dx[1]
end

function ys(grid::G) where {G <: Grid}
    return grid.o[2] .+ (0:(grid.N[2]-1)) * grid.dx[2]
end

@inline function pos2ind(grid::G, pos) where {G <: Grid}
    return CartesianIndex( (1 + Int( floor( (pos[i] - grid.o[i]) / grid.dx[i] )) for i=1:2 )...)
        # CartesianIndex(1, 1) + CartesianIndex(Int.(round.((pos - grid.o) ./ grid.dx))...)
end

@inline function ind2pos(grid::G, ind) where {G <: Grid}
    return @SVector [grid.o[i] + (ind[i] - 1) * grid.dx[i] for i=1:2]
end

function fill!(f, grid, M)
    for ind in CartesianIndices(M)
        pos = ind2pos(grid, ind)
        M[ind] = f(pos)
    end
end

@inline normsq(x) = mapreduce(abs2, sum, x)

@inline hk(k) = k==0 ? 1 : 1/sqrt(2)
@inline hk(k1, k2) = hk(k1) * hk(k2)

@inline Λ(k; d=length(k), s=(1+d)/2)  = (1 + normsq(k))^(-s)

@inline Λ2D(k1, k2)  = (1 + k1^2 + k2^2)^(-1.5)


function dct_map(grid::G, M) where {G <: Grid}

    @assert size(M) == grid.N
    
    N1, N2 = grid.N
    L1, L2 = lengths(grid)

    normalize!(M)

    # do the un-normalized DCT
    Y = grid.dct_plan * M


    # do the normalization
    δ = (L1/N1) * (L2/N2) / (2^2)
    for k1=0:(N1-1), k2=0:(N2-1)
        Y[k1+1, k2+1] *= δ
        if k1==0 || k2==0
            Y[k1+1,k2+1] = Y[k1+1,k2+1]  / hk(k1,k2)
        end
    end
    
    return Y

end

function dct_trajectory(grid, traj)

    M = zeros(grid.N)

    N = length(traj)

    for p in traj
        ind = pos2ind(grid, p)
        if ind in CartesianIndices(M)
            M[ind] += 1/N
        end
    end

    return dct_map(grid, M)
end

function normalize!(M)
    s = sum(M)
    M .= M / s
end


function ergodicity(grid::G, M1::MT, M2::MT) where {G <: Grid, T, MT <: Matrix{T}}
    size(M1) == size(M2) || throw(DimensionMismatch())
    N1, N2 = size(M1)
    normalize!(M1)
    normalize!(M2)

    M1k = dct_map(grid, M1)
    M2k = dct_map(grid, M2)
    
    d = 0.0
    for ind in CartesianIndices(M1k)
        k1 = ind[1]
        k2 = ind[2]
        d += Λ2D(k1-1, k2-1) * (M1k[ind] - M2k[ind])^2
    end
    return sqrt(d) / prod(grid.dx)
end

function ergodicity(grid, M1::MT, traj::VP) where {T, P, MT <: Matrix{T}, VP <: AbstractVector{P}}

    size(M1) == grid.N || throw(DimensionMismatch())
    
    M2 = zeros(grid.N)

    N = length(traj)

    for p in traj
        ind = pos2ind(grid, p)
        if ind in CartesianIndices(M2)
            M2[ind] += 1/N
        end
    end

    return ergodicity(grid, M1, M2)

end


function grad_fk(grid, p, k)
    
    L1, L2 = lengths(grid)
    p1, p2 = (p - grid.o)
    k1, k2 = k
    
    return (1/(hk(k1, k2) * prod(grid.dx)))  * (@SVector [
        -k1 * sin(π * k1 * p1 / L1) * cos(π * k2 * p2 / L2),
        -k2 * cos(π * k1 * p1 / L1) * sin(π * k2 * p2 / L2),
    ]) 
end

"""
    computes ck - Mk
"""
function ckMk(grid, traj, M)
    # grid == ergo_grid
    # M == target_spatial_data
    size(M) == grid.N || throw(DimensionMismatch())

    normalize!(M)

    Mk = dct_map(grid, M)
    # println("Mk: $(Mk)")
    ck = dct_trajectory(grid, traj)

    return ck - Mk
end

# this is a direction vector for single-integrator control
function ergodic_descent_direction(grid, p, traj, M)

    ck_minus_Mk = ckMk(grid, traj, M)
    return ergodic_descent_direction(grid, p, ck_minus_Mk)

end

function ergodic_descent_direction(grid, p, ck_minus_Mk)

    size(ck_minus_Mk) == grid.N || throw(DimensionMismatch())
    
    N1, N2 = grid.N
    
    bx = 0;
    by = 0; #  = [ 0, 0.]
    for i=1:N1, j=1:N2
        k = @SVector [i-1, j-1]
        k1, k2 = k
        bk = (Λ2D(k1, k2) * ck_minus_Mk[i,j]) * grad_fk(grid, p, k)
        bx += bk[1]
        by += bk[2]
    end
    return @SVector [bx, by]
end



function convex_bounary_correction(convex_polygon, p, u; speed_max = 1.8, min_safe_d = 0.2)

    # Compute the closest point on the boundary of the convex shape to the robot pos p
    distance, closest_point = ConvexBoundAvoidance.minimum_distance_to_boundary(convex_polygon, p)

    # Compute a normal unit vector pointing towards the centroid of the convex shape 
    normal_vector = ConvexBoundAvoidance.normal_vector_to_centroid(convex_polygon, closest_point)

    # Compute the force field 
    uff = normal_vector * speed_max
    # Compute the adjusted velocity vector
    u_cmd = min(1, (distance/min_safe_d)) * u + max(0, (1 - (distance/min_safe_d)))*uff
    
    # debugging
    # println("u_ergo: $(u)")
    # println("speed: $(speed)")
    # pritnln("field force input: $(uff)")
    # pritnln("commanded velocity: $(u_cmd)")

    return u_cmd
end


function boundary_correction(grid, p, u; α=1)

    # decompose the control input
    u1, u2 = u

    lower = grid.o
    upper = grid.o + lengths(grid)
        
    # h1 constraint
    # h1dot(x, u) ≥ -α(h1)
    # ∴ u ≥ - α * h1
    h1 = p[1] - lower[1]
    ux_min = -α * h1

    # h2 constraint
    # h2dot(x, u) ≥ -α(h2)
    # -u ≥  -α * h2
    # u ≤ α * h2
    h2 = upper[1] - p[1]
    ux_max = α * h2

    # h3 constraint
    # h3dot(x, u) ≥ -α(h3)
    # ∴ u ≥ - α * h3
    h3 = p[2] - lower[2]
    uy_min = -α * h3

    # h2 constraint
    # h2dot(x, u) ≥ -α(h2)
    # -u ≥  -α * h2
    # u ≤ α * h2
    h4 = upper[2] - p[2]
    uy_max = α * h4

    # do the corrections

    ux = clamp(u1, ux_min, ux_max)
    uy = clamp(u2, uy_min, uy_max)
    
    

    return @SVector [ux, uy]
end


function boundary_correction_discrete_time(grid, p, u; γ=0.5, ΔT)


    # decompose the control input
    u1, u2 = u

    lower = grid.o
    upper = grid.o + lengths(grid)

    # h_{k+1} - h_k >= - α * h[k] 
    # h_{k+1} >= γ * h[k]

        
    # h1 constraint
    d1 = p[1] - lower[1]
    d1k = γ * d1
    ux_min = - d1k / ΔT

    # h2 constraint
    d2 = upper[1] - p[1]
    d2k = γ * d2
    ux_max = d2k / ΔT

    # h3 constraint
    d3 = p[2] - lower[2]
    d3k = γ * d3
    uy_min = -d3k / ΔT

    # h4 constraint
    d4 = upper[2] - p[2]
    d4k = γ * d4
    uy_max = d4k / ΔT

    # do the corrections
    ux = clamp(u1, ux_min, ux_max)
    uy = clamp(u2, uy_min, uy_max)
    
    return @SVector [ux, uy]
end


function controller_single_integrator_cvx_bound(grid, p, traj, M, convex_polygon; umax=1.0, do_boundary_correction=true)
    
    # println("M: $(M)")
    # println("here")
    b_ergo = ergodic_descent_direction(grid, p, traj, M)
    # println("b_ergo: $(b_ergo)")
    # println("b_ergo: $(b_ergo)")
    u_ergo = - umax * normalize(b_ergo)
    if do_boundary_correction
        # return boundary_correction_discrete_time(grid, p, u_ergo; ΔT)
        return convex_bounary_correction(convex_polygon, p, u_ergo; speed_max = umax)
    else
        return u_ergo
    end
     
end



function controller_single_integrator(grid, p, traj, M; umax=1.0, do_boundary_correction=true)
    
    # println("M: $(M)")
    b_ergo = ergodic_descent_direction(grid, p, traj, M)
    # println("b_ergo: $(b_ergo)")
    u_ergo = - umax * normalize(b_ergo)
    if do_boundary_correction
        # return boundary_correction_discrete_time(grid, p, u_ergo; ΔT)
        return boundary_correction(grid, p, u_ergo)
    else
        return u_ergo
    end
     
end


function controller_single_integrator(grid, p, ck_minus_Mk; umax=1.0, do_boundary_correction=true)
    
    b_ergo = ergodic_descent_direction(grid, p, ck_minus_Mk)
    u_ergo = - umax * normalize(b_ergo)
    if do_boundary_correction
        # return boundary_correction_discrete_time(grid, p, u_ergo; ΔT)
        return boundary_correction(grid, p, u_ergo)
    else
        return u_ergo
    end
     
end


@recipe function f(grid::G, M) where {G <: Grid}

    xs = grid.o[1] .+ grid.dx[1] * (0:(grid.N[1]))
    ys = grid.o[2] .+ grid.dx[2] * (0:(grid.N[2]))

    # @show xs, ys

    @series begin
        seriestype --> :heatmap
        xlabel --> "x"
        ylabel --> "y"
        aspect_ratio --> :equal
        xs, ys, M'
    end

end



end