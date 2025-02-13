using DifferentialEquations, Optimization, OptimizationOptimJL, SciMLBase

# Define the system dynamics
dynamics!(du, u, p, t) = begin
    x, b = u
    α, r_func, u_func = p
    du[1] = u_func(t)  # Control input u(t)
    du[2] = -α * abs(du[1])^3 + r_func(t)
end

# Define the cost function
function ergodic_metric(u, p)
    λk, ck, ĉk = p
    return mapreduce(t -> ergodic_metric(sol(t), (λk, ck, ĉk)), +, sol.t)
end

# Define the objective function
function objective(control, p)
    T, u_max, α, r_func, λk, ck, ĉk = p
    u_func = t -> SciMLBase.scalarize(control(t)[1])
    prob = ODEProblem(dynamics!, [0.0, p[8]], (0, T), (α, r_func, u_func))
    sol = solve(prob, Tsit5())
    return sum(ergodic_metric(sol(t), (λk, ck, ĉk)) for t in sol.t)
end

# Define constraints
function constraints(control, p)
    T, u_max, α, r_func, λk, ck, ĉk = p
    u_func = t -> SciMLBase.scalarize(control(t)[1])
    prob = ODEProblem(dynamics!, [0.0, p[8]], (0, T), (α, r_func, u_func))
    sol = solve(prob, Tsit5())
    return [maximum(abs.(u_func.(LinRange(0, T, 100)))) - u_max, sol(T, idxs=2) - 0] # b(T) = 0
end

# Define initial conditions and parameters
α = 1.0  # Coefficient in dynamics
r_func = t -> 0.1  # Example replenishment function
λk = [1.0]  # Example weights
ck, ĉk = [0.5], [0.3]  # Example coefficients
u_max = 1.0  # Control constraint

# Initialize dataset storage
results = []

# Sweep through various values of b0 and T
for b0 in 0.5:0.5:2.0, T in 5.0:5.0:20.0
    p = (T, u_max, α, r_func, λk, ck, ĉk, b0)
    control_guess = t -> [0.0]  # Initial guess for control
    opt_prob = OptimizationFunction(objective, Optimization.AutoForwardDiff(); constraints=constraints)
    prob = OptimizationProblem(opt_prob, control_guess, p, constraints=constraints)
    sol = solve(prob, Optim.BFGS())
    push!(results, (b0, T, sol.minimum))  # Store information deficit
end

println("Dataset of information deficit as a function of b0 and T:", results)
