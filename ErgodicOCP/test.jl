using DifferentialEquations, JuMP, Ipopt, FFTW, Plots

# Robot dynamics with cubic velocity-dependent discharge rate
function robot_dynamics!(du, u, p, t)
    x, y, soc = u  # Robot position (x, y) and energy SOC
    vx, vy = p      # Control inputs (velocities)
    
    α, r_func = p[3], p[4]  # Energy consumption rate and solar function
    
    du[1] = vx  # dx/dt = velocity_x
    du[2] = vy  # dy/dt = velocity_y
    du[3] = -α * (vx^2 + vy^2)^(3/2) + r_func(t)  # dSOC/dt
end

# Solar charging model (diurnal)
function solar_charging(t)
    return max(0, sin(π * (t % 24) / 12))  # Peaks at noon
end

# Ergodic metric calculation
function ergodic_metric(traj, phi, K=10)
    c_k = zeros(K, K)
    phi_k = zeros(K, K)
    λ_k = zeros(K,K);
    
    for i in 1:K, j in 1:K
        λ_k[i,j] = 1 / (1 + i^2 + j^2)  # Weighting factor

        println("traj: ", traj[:,1])
        
        # Compute trajectory density
        c_k[i, j] = sum(exp(-((traj[:,1] .- i/K).^2 + (traj[:,2] .- j/K).^2)))
        
        # Compute information density Fourier components
        phi_k[i, j] = exp(-((phi[1] - i/K)^2 + (phi[2] - j/K)^2))
    end
    
    return sum(λ_k .* (c_k - phi_k).^2)
end

# Simulate the system
function simulate_trajectory(u0, p, T)
    tspan = (0.0, T)
    prob = ODEProblem(robot_dynamics!, u0, tspan, p)
    sol = solve(prob, Tsit5())
    return sol
end

# Optimize the trajectory
function optimize_trajectory(u0, p, T, N=50)
    model = Model(Ipopt.Optimizer)
    
    @variable(model, u[1:N, 1:2])  # Velocity at each time step
    dt = T / N
    x, y, soc = u0
    
    @variable(model, x_traj[1:N])
    @variable(model, y_traj[1:N])
    @variable(model, soc_traj[1:N])
    
    @constraint(model, x_traj[1] == x)
    @constraint(model, y_traj[1] == y)
    @constraint(model, soc_traj[1] == soc)
    
    α, r_func = p[1], p[2]
    
    for t in 1:N-1
        @constraint(model, x_traj[t+1] == x_traj[t] + dt * u[t,1])
        @constraint(model, y_traj[t+1] == y_traj[t] + dt * u[t,2])
        @constraint(model, soc_traj[t+1] == soc_traj[t] - α * (u[t,1]^2 + u[t,2]^2)^(3/2) * dt + r_func(t*dt) * dt)
    end
    
    @constraint(model, soc_traj .≥ 0)
    
    phi = [0.5, 0.5]  # Target information density
    @objective(model, Min, ergodic_metric([x_traj y_traj], phi))
    
    optimize!(model)
    
    return value.(u), value.(x_traj), value.(y_traj), value.(soc_traj)
end

# Run optimization and plot results
u0 = [0.0, 0.0, 1.0]  # Initial state (x, y, SOC)
params = (0.1, solar_charging)  # (α, solar function)
T = 24  # Time horizon

u_opt, x_traj, y_traj, soc_traj = optimize_trajectory(u0, params, T)

plot(x_traj, y_traj, label="Optimal Path")
scatter!([0.5], [0.5], label="Target Info Density", color=:red)
xlabel!("X Position")
ylabel!("Y Position")

plot(soc_traj, label="SOC", color=:green)
xlabel!("Time")
ylabel!("State of Charge (SOC)")
