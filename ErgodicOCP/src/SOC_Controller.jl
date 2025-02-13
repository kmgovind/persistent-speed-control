module SoCController

include("solar_insolation.jl")
using .SolarInsolationModel


# Vehicle Parameters
Base.@kwdef struct ASV_Params
    b_max::Float32 = 6500; # max soc in Wh
    b_min::Float32 = 0; # min soc in Wh
    panel_area::Float32 = 4; # m^2
    panel_efficiency::Float32 = 0.25; # 25% panel efficiency
    v_max::Float32 = 2.315; # max boat speed in m/s 
    v_min::Float32 = 0; # min boat speed in m/s

    k_h::Float32 = 10; # Hotel Load
    k_m::Float32 = 83; # Motor multiplier, need to tune
end

boat = ASV_Params();

# Environment Parameters
dayOfYear = 288; # corresponds to October 15th
lat = 35.45; # degrees - corresponds to Jordan Lake
# Δt = 0.1; # time step in hours
# t = 10:Δt:13;
# og_time = t;
# t = t .+ 12;
# t = t .% 24; # time over a day from noon to noon
# n = length(t);

# Compute the lower Barrier

# ============================================================
function compute_lcbf(t, Δt)
    n = length(t);
    og_time = t;
    ϵ₋ = zeros(n); # energy deficit
    Ps = zeros(n);
    for i = 1:n
        Ps[i] = max(0, SolarInsolation(dayOfYear, t[i], lat))*1000* boat.panel_area * boat.panel_efficiency;
        ϵ₋[i] = boat.k_h*(og_time[i] - og_time[1]) - sum(Ps[1:i]*Δt);
    end

    lcbf = zeros(n);
    
    for i = 1:n
        ϵ₋dag = ϵ₋ .- ϵ₋[i];
        lcbf[i] = max(0, maximum(ϵ₋dag[i:end]));
    end

    return lcbf
end

function compute_ucbf(t, Δt)
    n = length(t);
    og_time = t;
    ϵ₊ = zeros(n); # energy surplus
    Ps = zeros(n);
    for i = 1:n
        Ps[i] = max(0, SolarInsolation(dayOfYear, t[i], lat))*1000* boat.panel_area * boat.panel_efficiency;
        # boat.k_h*(og_time[i] - og_time[1])
        ϵ₊[i] = sum(Ps[1:i]*Δt) - (boat.k_h + boat.k_m*(boat.v_max^3))*(og_time[i] - og_time[1]);
    end

    ucbf = zeros(n);

    for i = 1:n
        ϵ₊dag = ϵ₊ .- ϵ₊[i];
        ucbf[i] = boat.b_max - max(0, maximum(ϵ₊dag[i:end]));
    end

    return ucbf
end


function batterymodel!(boat, dayOfYear, time, lat, vel, soc, dt)
    # Solar Insolation returns in kW
    p_in = max(0,SolarInsolation(dayOfYear, time, lat))* 1000 * boat.panel_area * boat.panel_efficiency;
    p_out = boat.k_h + boat.k_m * (vel^3);
    soc_est = soc + (p_in - p_out)*dt; # power update in Wh
    soc_est = min(soc_est, boat.b_max); # cap charge at soc_max
    return soc_est;
end

function powermodel!(boat, dayOfYear, time, lat, vel, soc, dt)
    # Solar Insolation returns in kW
    p_in = max(0,SolarInsolation(dayOfYear, time, lat))* 1000 * boat.panel_area * boat.panel_efficiency;
    p_out = boat.k_h + boat.k_m * (vel^3);
    out = p_in - p_out;
    return out;
end

function zeropower!(boat, dayOfYear, time, lat, soc, dt)
    p_in = max(0,SolarInsolation(dayOfYear, time, lat))* 1000 * boat.panel_area * boat.panel_efficiency;
    if p_in < boat.k_h
        vel = 0;
    else
        vel = cbrt((p_in - boat.k_h)/boat.k_m);
    end
    return vel;
end


function generate_SOC_target(lcbf, ucbf, soc_begin, soc_target, t, Δt)
    n = length(t);

    # Learning gains
    k_p = -5e-5; # Learning P gain (-1e-5)
    k_d = -1e-5; # Learning D gain 5e-5 -1e-5


    xmax = 0; # best distance travel
    pstar = 0;
    num_iters = 5000;
    p_list = zeros(num_iters);
    p2_list = zeros(n);
    error_list = zeros(num_iters);
    error_rate = zeros(num_iters);
    p_list[1] = 0.5; # Initial guess for p2
    p2min = 1/(3*boat.k_m*(boat.v_max^2));

    δ = 150; # 50Wh barrier on lcbf
    lcbf_dot = diff(lcbf); # Derivative of lcbf

    x = zeros(n);
    b = ones(n)*soc_begin;
    v = ones(n)*boat.v_max;
    old_b = zeros(n);

    for day = 1:1:num_iters
        # println("xmax:$(xmax)")
        x = zeros(n);
        b = ones(n)*soc_begin;
        v = ones(n)*boat.v_max;

        p2 = p_list[day];

        for j in 2:n
            i = j-1;
            if i == 1 # initial conditions
                x[i] = 0;
                b[i] = soc_begin;
            end

            p2_list[i] = p2;

            # println(p2);
            # Compute unconstrained velocity and SOC
            v[i] = sqrt(1/(3 * p2 * boat.k_m)); # removed negative sign from numerator to let p be positive
            b_dot = powermodel!(boat, dayOfYear, t[i], lat, v[i], b[i], Δt);

            # Impose Boundary Conditions
            if b[i] <= lcbf[i]
                v[i] = 0;
            elseif 0 < (b[i] - lcbf[i]) < δ
                v[i] = (b[i] - lcbf[i])/δ * v[i] + (1 - (b[i] - lcbf[i])/δ) * boat.v_min;
            elseif b[i] >= ucbf[i]
                v[i] = boat.v_max;
            elseif 0 < (ucbf[i] - b[i]) < δ
                v[i] = (ucbf[i] - b[i])/δ * v[i] + (1 - (ucbf[i] - b[i])/δ) * boat.v_max;
            end

            # Move boat
            x[j] = x[i] + (v[i] * 60 * 60) * Δt;
            b[j] = batterymodel!(boat, dayOfYear, t[i], lat, v[i], b[i], Δt);
            v[j] = v[i];

            # Continuous rate update
            if day != 1
                cont_error = b[j] - old_b[j];
                # pbar = log10(p2) + k_d * cont_error;
                pbar = log10(p_list[day-1]) + k_d * cont_error; # update co-state from previous day
                p2 = 10^pbar;
                # println("Day: ", day, "\t Time: ", j, "\t new p2: ", p2);
            end
        end

        # Learning adjustment
        if day < num_iters
            error_rate[day] = 0;
            error_list[day] = b[end] - soc_target;
            if day != 1
                error_rate[day] = error_list[day] - error_list[day-1];
            end
            pbar = log10(p_list[day]) + k_p * error_list[day];
            p_list[day+1] = 10^pbar;
        end

        old_b = b;

        # Find optimal p2 and x
        if x[end] > xmax
            xmax = x[end];
            pstar = p2;
        end

        if abs(b[end] - soc_target) < 500
            break;
        end
    end
    return b
end


function generate_vel_profile(lcbf, ucbf, b, t, Δt)
    n = length(t);

    # Store the target profile
    soc_profile = b;
    soc_kp = 0.01; 
    soc_ki = 0.0001;
    soc_kd = 0.001;

    # Create variables to store the ASV's state in this simulation
    b_sim = ones(n);
    v_sim = ones(n)*boat.v_max;
    e_sim = zeros(n); # error profile
    δ = 150; # 50Wh barrier on lcbf

    for j in 2:n
        i = j-1;
        if i == 1 # initial conditions
            b_sim[i] = 1;
        end

        e_sim[i] = b_sim[i] - soc_profile[i];

        # Compute unconstrained velocity and SOC
        global v_sim[i] = soc_kp * e_sim[i] + soc_ki * sum(e_sim) + soc_kd * diff(e_sim)[i];
        global v_sim[i] = max(0, min(v_sim[i], boat.v_max));
        
        # Impose Boundary Conditions
        if b_sim[i] <= lcbf[i]
            global v_sim[i] = 0;
        elseif 0 < (b_sim[i] - lcbf[i]) < δ
            global v_sim[i] = (b_sim[i] - lcbf[i])/δ * v_sim[i] + (1 - (b_sim[i] - lcbf[i])/δ) * boat.v_min;
        elseif b_sim[i] >= ucbf[i]
            global v_sim[i] = boat.v_max;
        elseif 0 < (ucbf[i] - b_sim[i]) < δ
            global v_sim[i] = (ucbf[i] - b_sim[i])/δ * v_sim[i] + (1 - (ucbf[i] - b_sim[i])/δ) * boat.v_max;
        end

        # Move boat
        b_sim[j] = batterymodel!(boat, dayOfYear, t[i], lat, v_sim[i], b_sim[i], Δt);
        v_sim[j] = v_sim[i];
    end

    return v_sim
end

function generate_vel_profile(t, Δt, soc_begin, soc_target)
    lcbf = compute_lcbf(t, Δt) 
    ucbf = compute_ucbf(t, Δt)
    # println("computed ucbf")
    b = generate_SOC_target(lcbf, ucbf, soc_begin, soc_target, t, Δt)
    # println("computed b")
    v = generate_vel_profile(lcbf, ucbf, b)
    return lcbf, ucbf, b, v
end


"""
Real-time PID controller for speed control.
...
# Arguments
- `current_soc::Float64`: the current state of charge of the ASV.
- `target_soc::Float64`: the goal state of charge of the ASV at the computation timestep.
- `error_sum::Float64`: the summation of error over the mission.
- `error::Float64`: the error from the previous computation.
...
"""
function speed_controller(current_soc, target_soc, error_sum, error)
    # PID Gains
    kp = 0.01; 
    ki = 0.0001;
    kd = 0.001;

    # PID
    prev_error = error;
    error = current_soc - target_soc;
    error_sum += error;
    difference = error - prev_error;
    speed = kp*error + ki*error_sum + kd*difference;
    speed = max(0, min(speed, boat.v_max));

    return speed, error_sum, error
end

# Target SOC based PID speed controller - using an error vector
function speed_controller(current_soc, target_soc, error)
    # PID Gains
    kp = 0.01; 
    ki = 0.0001;
    kd = 0.001;

    # PID
    push!(error, current_soc - target_soc);
    difference = error[end] - error[end-1];
    speed = kp*error[end] + ki*sum(error) + kd*difference;
    speed = max(0, min(speed, boat.v_max));

    return speed, error
end


end