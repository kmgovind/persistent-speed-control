module Variograms

using LinearAlgebra, LsqFit, StatsBase, DataFrames

# Function to calculate pairwise distances
function pairwise_distances(measurements)
    n = length(measurements)
    h = zeros(n, n)  # Spatial distances
    γ = zeros(n, n)  # Semi-variogram

    for i in 1:n
        for j in 1:n
            h[i, j] = sqrt((measurements[i].p[1] - measurements[j].p[1])^2 + (measurements[i].p[2] - measurements[j].p[2])^2)
            γ[i, j] = 0.5 * (measurements[i].y - measurements[j].y)^2
        end
    end
    return h, γ
end

# Function to calculate the empirical variogram
function empirical_variogram(h, γ, n_bins=50)
    # h_max = maximum(h)
    h_max = 30
    h_edges = range(0, h_max, length=n_bins+1)
    
    bins = DataFrame(h_mid = Float64[], variogram = Float64[], count = Int64[])
    
    for i in 1:n_bins
        h_bin = h_edges[i] .<= h .< h_edges[i+1]
        bin_count = count(h_bin)
        if bin_count > 0
            h_mid = (h_edges[i] + h_edges[i+1]) / 2
            variogram_value = mean(γ[h_bin])
            push!(bins, (h_mid, variogram_value, bin_count))
            # push the data all together at the end rather than on each loop
        end
    end

    return bins
end

function covariance(x1, x2, λₓ, σ)
    return (σ^2) .* exp(-norm.(x1.-x2)/λₓ);
end

function matern12_variogram(h, σ_sq, λₓ)
    return σ_sq .* (1 .- exp.(-h./λₓ))
end

function matern12_variogram(h, p)
    return p[1] .* (1 .- exp.(-h./p[2]))
end

function matern12_log(h, σ_sq, λₓ)
    return log(σ_sq).-(h./λₓ)
    # return -(1/λₓ).*h
end

function matern12_lin(h, σ_sq, λₓ)
    return -1/λₓ
end

# Recursive least squares fitting for the log variogram provided variance
function rls_weight(emp_vario, initial_params, λ)
    n = length(emp_vario.h_mid)
    params = initial_params

    half_n = Int(floor(n/2));

    # RLS udpate of λₓ (Only base this off of first half of bins)
    P = 2
    for i in 1:half_n
        h_i = emp_vario.h_mid[i]
        γ_i = log.(sign.(emp_vario.variogram[i] - params[1]) .* (emp_vario.variogram[i] - params[1]))

        # Jacobian
        # phi_i = [matern32_variogram(h_i, 1, params[2]) / params[1],   # Derivative wrt σ²
        #          matern32_variogram(h_i, params[1], 1) / params[2]]   # Derivative wrt ℓ

        phi_i = h_i/(params[2]^2)

        # Prediction error
        # γ_hat_i = matern12_variogram(h_i, params[1], params[2])
        γ_hat_i = matern12_log(h_i, params[1], params[2])
        e_i = γ_i - γ_hat_i

        # Kalman gain
        K = P * phi_i / (λ + phi_i * P * phi_i)

        # Update parameters
        params[2] = max(0.01, params[2] + K * e_i)

        # Update covariance matrix
        P = (I - K * phi_i') * P / λ
    end

    # RLS update of σ_sq (weight towards last half of bins)
    P = 2
    for i in 1:n
        h_i = emp_vario.h_mid[i]
        γ_i = emp_vario.variogram[i]

        # Jacobian
        # phi_i = [matern32_variogram(h_i, 1, params[2]) / params[1],   # Derivative wrt σ²
        #          matern32_variogram(h_i, params[1], 1) / params[2]]   # Derivative wrt ℓ

        phi_i = 1 - exp(-h_i/params[2])

        # Prediction error
        γ_hat_i = matern12_variogram(h_i, params[1], params[2])
        e_i = γ_i - γ_hat_i

        # Kalman gain
        K = P * phi_i / (λ + phi_i * P * phi_i)

        # Update parameters
        params[1] = max(0.01, params[1] + K * e_i)

        # Update covariance matrix
        P = (I - K * phi_i') * P / λ
    end
    

    return params
end

"Fit hyperparameters to measurements"
function hp_fit(measurements)
    h, γ = pairwise_distances(measurements);
    emp_vario = empirical_variogram(h, γ);
    param_fit = curve_fit(matern12_variogram, emp_vario.h_mid, emp_vario.variogram, [1.0,1.0]);
    return param_fit.param[1], param_fit.param[2]
end

"Fit hyperparameters to spatiotemporal measurements"
function hp_fit(measurements)
    h, u, γ = pairwise_distances_st(measurements);
    emp_vario = empirical_variogram_st(h, u, γ);
    lags = [emp_vario.h_mid emp_vario.u_mid]
    param_fit = curve_fit(matern12_variogram, lags, emp_vario.variogram, [1.0,2.0,5.0])
    return param_fit.param[1], param_fit.param[2], param_fit.param[3]
end

# Function to calculate pairwise distances
function pairwise_distances_st(measurements)
    n = length(measurements)
    h = zeros(n, n)  # Spatial distances
    u = zeros(n, n)  # Temporal distances
    γ = zeros(n, n)  # Semi-variogram

    for i in 1:n
        for j in 1:n
            h[i, j] = sqrt((measurements[i].p[1] - measurements[j].p[1])^2 + (measurements[i].p[2] - measurements[j].p[2])^2)
            u[i, j] = abs(measurements[i].t - measurements[j].t)
            γ[i, j] = 0.5 * (measurements[i].y - measurements[j].y)^2
        end
    end
    return h, u, γ
end

# Function to calculate the empirical variogram
function empirical_variogram_st(h, u, γ, n_bins=50)
    h_max = maximum(h)
    u_max = maximum(u)
    h_edges = range(0, h_max, length=n_bins+1)
    u_edges = range(0, u_max, length=n_bins+1)
    
    bins = DataFrame(h_mid = Float64[], u_mid = Float64[], variogram = Float64[], count = Int64[])
    
    for i in 1:n_bins
        for j in 1:n_bins
            h_bin = h_edges[i] .<= h .< h_edges[i+1]
            u_bin = u_edges[j] .<= u .< u_edges[j+1]
            mask = h_bin .& u_bin
            bin_count = count(mask)
            if bin_count > 0
                h_mid = (h_edges[i] + h_edges[i+1]) / 2
                u_mid = (u_edges[j] + u_edges[j+1]) / 2
                variogram_value = mean(γ[mask])
                push!(bins, (h_mid, u_mid, variogram_value, bin_count))
            end
        end
    end
    return bins
end

function matern12_variogram(h, u, σ_sq, λₓ, λₜ)
    return σ_sq .* (1 .- exp.(-h./λₓ) .* exp.(-u./λₜ))
end

function matern12_variogram(lags, p)
    h = lags[:,1]
    u = lags[:,2]
    return p[1] .* (1 .- exp.(-h./p[2]).* exp.(-u./p[3]))
end


end