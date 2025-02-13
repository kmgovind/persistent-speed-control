module STGPKF # spatiotemporal-Gaussian-process Kalman filter

using LinearAlgebra

using ..JordanLakeDomain

M = length(𝐬)

mutable struct KeyParameters
    λₛ::Float64 # 1 / km
    σ::Float64
    λₜ::Float64 # 1 / s
    σᵣ::Float64
    const Δt::Int64
    𝐊ₘ::Matrix{Float64}
    𝐊ₘʰᵃˡᶠ::Matrix{Float64}
    𝐊ₘ⁻¹::Matrix{Float64}
    Σ₀::Float64
    F̄::Float64
    𝐀::Matrix{Float64}
    𝐇ᵇᵃʳ::Matrix{Float64}
    𝐐::Matrix{Float64}

    KeyParameters(λₛ, σ, λₜ, σᵣ, Δt,
        𝐊ₘ=reshape([exp(-λₛ * norm(𝐬[j] - 𝐬[i])) for i = 1:M for j = 1:M], (M, M)),
        𝐊ₘʰᵃˡᶠ=cholesky(𝐊ₘ).L,
        𝐊ₘ⁻¹=inv(𝐊ₘ),
        Σ₀=1 / (2 * λₜ),
        F̄=exp(-λₜ * Δt),
        𝐀=Diagonal(F̄ * ones(M)),
        𝐇ᵇᵃʳ=Diagonal(σ * sqrt(2 * λₜ) * ones(M)),
        𝐐=Diagonal(1 / (2 * λₜ) * (1 - exp(-2 * λₜ * Δt)) * ones(M))
    ) =
        new(λₛ, σ, λₜ, σᵣ, Δt, 𝐊ₘ, 𝐊ₘʰᵃˡᶠ, 𝐊ₘ⁻¹, Σ₀, F̄, 𝐀, 𝐇ᵇᵃʳ, 𝐐)
end

mutable struct Estimator
    𝐯ʰᵃᵗⱼₗᵢ::Matrix{Float64} # 𝐯ʰᵃᵗⱼₗᵢ[:, 1] is 𝐯ʰᵃᵗᵢₗᵢ and 𝐯ʰᵃᵗⱼₗᵢ[:, 2] is 𝐯ʰᵃᵗ₍ᵢ₊₁₎ₗᵢ
    𝚺ⱼₗᵢ::Array{Float64,3} # 𝚺ⱼₗᵢ[:, :, 1] is 𝚺ᵢₗᵢ and 𝚺ⱼₗᵢ[:, 2] is 𝚺ᵢ₊₁ₗᵢ
    𝐱ʰᵃᵗⱼₗᵢ::Matrix{Float64} # 𝐱ʰᵃᵗⱼₗᵢ[:, 1] is 𝐱ʰᵃᵗᵢₗᵢ and 𝐱ʰᵃᵗⱼₗᵢ[:, 2] is 𝐱ʰᵃᵗ₍ᵢ₊₁₎ₗᵢ
    𝚺ⱼₗᵢ_𝐱ʰᵃᵗ::Array{Float64,3} # error covariance matrices for 𝐱ʰᵃᵗ

    Estimator(𝐯ʰᵃᵗⱼₗᵢ=zeros(M, 2), 𝚺ⱼₗᵢ=zeros(M, M, 2),
        𝐱ʰᵃᵗⱼₗᵢ=zeros(M, 2),
        𝚺ⱼₗᵢ_𝐱ʰᵃᵗ=zeros(M, M, 2)) = new(𝐯ʰᵃᵗⱼₗᵢ, 𝚺ⱼₗᵢ, 𝐱ʰᵃᵗⱼₗᵢ, 𝚺ⱼₗᵢ_𝐱ʰᵃᵗ)
end

function initialize!(estimator::Estimator, key_parameters::KeyParameters)
    # Update
    estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1] = zeros(M)
    estimator.𝚺ⱼₗᵢ[:, :, 1] = Diagonal(key_parameters.Σ₀ * ones(M))
    estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1] = (key_parameters.𝐊ₘʰᵃˡᶠ * key_parameters.𝐇ᵇᵃʳ
                               * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1])
    estimator.𝚺ⱼₗᵢ_𝐱ʰᵃᵗ[:, :, 1] = ((key_parameters.𝐊ₘʰᵃˡᶠ * key_parameters.𝐇ᵇᵃʳ)
                                    * estimator.𝚺ⱼₗᵢ[:, :, 1]
                                    * transpose(key_parameters.𝐊ₘʰᵃˡᶠ
                                                *
                                                key_parameters.𝐇ᵇᵃʳ))
    # Predict
    estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2] = key_parameters.𝐀 * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1]
    estimator.𝚺ⱼₗᵢ[:, :, 2] = (key_parameters.𝐀 * estimator.𝚺ⱼₗᵢ[:, :, 1]
                               * transpose(key_parameters.𝐀)
                               +
                               key_parameters.𝐐)
    estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 2] = (key_parameters.𝐊ₘʰᵃˡᶠ * key_parameters.𝐇ᵇᵃʳ
                               * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2])
    estimator.𝚺ⱼₗᵢ_𝐱ʰᵃᵗ[:, :, 2] = ((key_parameters.𝐊ₘʰᵃˡᶠ * key_parameters.𝐇ᵇᵃʳ)
                                    * estimator.𝚺ⱼₗᵢ[:, :, 2]
                                    * transpose(key_parameters.𝐊ₘʰᵃˡᶠ
                                                *
                                                key_parameters.𝐇ᵇᵃʳ))
end

function kₛ(key_parameters::KeyParameters, 𝐬₁, 𝐬₂)
    exp(-key_parameters.λₛ * norm(𝐬₁ - 𝐬₂))
end

function update_and_predict!(estimator::Estimator,
    key_parameters::KeyParameters, yᵢ::Float64, 𝐬ᵢᵗⁱˡᵈᵉ::Vector{Float64})
    # Update
    𝐇ᵢᵗⁱˡᵈᵉ = reshape([kₛ(key_parameters, 𝐬ᵢᵗⁱˡᵈᵉ, 𝐬[j]) for j = 1:M], (1, M)) * key_parameters.𝐊ₘ⁻¹
    𝐂ᵢ = 𝐇ᵢᵗⁱˡᵈᵉ * key_parameters.𝐊ₘʰᵃˡᶠ * key_parameters.𝐇ᵇᵃʳ
    𝐑ᵢ = Diagonal(key_parameters.σᵣ * ones(length(yᵢ)))
    𝐋ᵢ = (estimator.𝚺ⱼₗᵢ[:, :, 2] * transpose(𝐂ᵢ)
          * inv(𝐂ᵢ * estimator.𝚺ⱼₗᵢ[:, :, 2] * transpose(𝐂ᵢ) + 𝐑ᵢ))
    estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1] = (estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2]
                               +
                               𝐋ᵢ * (yᵢ - (𝐂ᵢ*estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2])[1]))
    estimator.𝚺ⱼₗᵢ[:, :, 1] = (estimator.𝚺ⱼₗᵢ[:, :, 2]
                               -
                               𝐋ᵢ * 𝐂ᵢ * estimator.𝚺ⱼₗᵢ[:, :, 2])
    estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1] = (key_parameters.𝐊ₘʰᵃˡᶠ
                               * key_parameters.𝐇ᵇᵃʳ
                               * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1])
    estimator.𝚺ⱼₗᵢ_𝐱ʰᵃᵗ[:, :, 1] = ((key_parameters.𝐊ₘʰᵃˡᶠ * key_parameters.𝐇ᵇᵃʳ)
                                    * estimator.𝚺ⱼₗᵢ[:, :, 1]
                                    * transpose(key_parameters.𝐊ₘʰᵃˡᶠ
                                                *
                                                key_parameters.𝐇ᵇᵃʳ))
    # Predict
    estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2] = key_parameters.𝐀 * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 1]
    estimator.𝚺ⱼₗᵢ[:, :, 2] = (key_parameters.𝐀 * estimator.𝚺ⱼₗᵢ[:, :, 1]
                               * transpose(key_parameters.𝐀)
                               +
                               key_parameters.𝐐)
    estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 2] = (key_parameters.𝐊ₘʰᵃˡᶠ
                               * key_parameters.𝐇ᵇᵃʳ
                               * estimator.𝐯ʰᵃᵗⱼₗᵢ[:, 2])
    estimator.𝚺ⱼₗᵢ_𝐱ʰᵃᵗ[:, :, 2] = ((key_parameters.𝐊ₘʰᵃˡᶠ * key_parameters.𝐇ᵇᵃʳ)
                                    * estimator.𝚺ⱼₗᵢ[:, :, 2]
                                    * transpose(key_parameters.𝐊ₘʰᵃˡᶠ
                                                *
                                                key_parameters.𝐇ᵇᵃʳ))
end

end
