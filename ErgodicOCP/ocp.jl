using JuMP, Ipopt, JLD2, ProgressLogging

# Define grid
xs = 0:0.1:10;
ys = 0:0.1:10;
dim_x, dim_y = length(xs), length(ys);

# Create clarity map and target clarity map
clarity_map = zeros(dim_x, dim_y);
target_clarity = zeros(dim_x, dim_y);

# Define clarity dynamics parameters
Q = 0.01  # Process noise (decay rate)
R = 0.1   # Measurement noise
C = 1.0   # Measurement strength
sensing_radius = 2.5  # Radius of the sensing area
alpha = 0.1  # Energy dissipation rate
umax = 2.35   # Maximum control input
T = 100       # Time horizon

# Define target clarity peaks
peaks = [(2.5, 2.5), (2.5, 7.5), (7.5, 2.5), (7.5, 7.5)]
for (px, py) in peaks
    for i in 1:dim_x
        for j in 1:dim_y
            target_clarity[i, j] += 0.9 * exp(-((xs[i] - px)^2 + (ys[j] - py)^2) / 4)
        end
    end
end

# Gaussian function for clarity update
gaussian(x, y, cx, cy, sigma) = exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2))

# Optimal control problem
model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "print_level", 5)  # Increase verbosity for debugging
set_optimizer_attribute(model, "tol", 1e-2)  # Set tolerance

@variable(model, x[1:T])
@variable(model, y[1:T])
@variable(model, u_x[1:T])
@variable(model, u_y[1:T])
@variable(model, b[1:T] >= 0)
@variable(model, clarity[1:dim_x, 1:dim_y, 1:T] >= 0)

# Initial conditions
@constraint(model, x[1] == 5.0)
@constraint(model, y[1] == 2.0)
@constraint(model, b[1] == 100.0)
@constraint(model, clarity[:, :, 1] .== clarity_map)

# Dynamics and clarity update
for t in 1:T-1
    @constraint(model, x[t+1] == x[t] + u_x[t])
    @constraint(model, y[t+1] == y[t] + u_y[t])
    @constraint(model, b[t+1] == b[t] - alpha * (u_x[t]^3 + u_y[t]^3) + R)
    
    for m in 1:dim_x, n in 1:dim_y
        weight = gaussian(xs[m], ys[n], x[t], y[t], sensing_radius / 2)
        @constraint(model, clarity[m, n, t+1] == clarity[m, n, t] + weight * ((C^2 / R) * (1 - clarity[m, n, t])^2 - Q * clarity[m, n, t]^2))
    end
end

# Control constraints
for t in 1:T
    @constraint(model, u_x[t]^2 + u_y[t]^2 <= umax^2)
end

# Objective: Minimize difference between target clarity and actual clarity
@objective(model, Min, sum((clarity[i, j, T] - target_clarity[i, j])^2 for i in 1:dim_x, j in 1:dim_y))

@time begin
    # Solve the optimization problem
    optimize!(model)
end

# Check solver status
if termination_status(model) == MOI.OPTIMAL
    println("Optimal solution found.")
else
    println("Solver did not find an optimal solution.")
    println("Termination status: ", termination_status(model))
    println("Primal status: ", primal_status(model))
    println("Dual status: ", dual_status(model))
end

# Retrieve optimal trajectory if solution is found
if termination_status(model) == MOI.OPTIMAL
    x_opt = value.(x)
    y_opt = value.(y)
    u_x_opt = value.(u_x)
    u_y_opt = value.(u_y)
    b_opt = value.(b)
    clarity_opt = value.(clarity[:, :, T])

    # Save results to a JLD2 file
    @save "single_ocp_results.jld2" x_opt y_opt u_x_opt u_y_opt b_opt clarity_opt target_clarity
end
