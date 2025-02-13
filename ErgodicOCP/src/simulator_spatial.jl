module SimulatorSpatial

using LinearAlgebra, StaticArrays, Interpolations
using ProgressLogging
using ..SyntheticData, ..NGPKF, ..ErgodicController, ..KF, ..SoCController, ..Variograms


# MATERN SPATIAL LENGTH SCALE = 1.0 km
# MATERN TEMPORAL LENGTH SCALE = 0.2 * 60 * 24 = 4.8 hrs

struct MeasurementSpatial{T, P, F}
  t::T
  p::P
  y::F
end

"""
  take_measurement(t, p, data; σ_meas = 0, Q_meas = σ_meas*I)

returns a SVector of the [wx, wy] at time t, and pos p by querying the data 
"""

function measure(t, p::SV, data::EDS; σ_meas=0, Q_meas=σ_meas * I) where {EDS<:EnvDataSpatial,SV<:SVector{2}}

  y = data(p...) + (σ_meas * randn(1))[1]
  return MeasurementSpatial(t, p, y)

end

function measure(t, ps::VSV, data::EDS; σ_meas=0, Q_meas = σ_meas * I) where {EDS<:EnvDataSpatial,SV<:SVector{2}, VSV <: AbstractVector{SV}}

  return [measure(t, p, data; Q_meas=Q_meas) for p in ps]

end

# function measure(t, p::SV, data::EDS; σ_meas=0, Q_meas=σ_meas * I) where {EDS<:EnvDataSpatial,SV<:SVector{2}}

#   y = data(p..., t) + Q_meas * randn(2)
#   return Measurement(t, p, y)

# end

# function measure(t, ps::VSV, data::EDS; σ_meas=0, Q_meas = σ_meas * I) where {EDS<:EnvDataSpatial,SV<:SVector{2}, VSV <: AbstractVector{SV}}

#   return [measure(t, p, data; Q_meas=Q_meas) for p in ps]

# end

function step(t, x::X, u::U, ΔT) where {X<:SVector,U<:SVector}

  A = I(2)
  B = ΔT * I(2)

  return A * x + B * u
end


function step(t, xs::XS, us::US, ΔT) where {X<:SVector,U<:SVector,XS<:AbstractVector{X},US<:AbstractVector{U}}

  length(xs) == length(us) || throw(DimensionMismatch())

  N = length(xs)

  return [step(t, xs[i], us[i], ΔT) for i = 1:N]

end


# struct SimResult{T,X,U,M,TV,W,EM}
#   ts::T
#   xs::X
#   us::U
#   measurements::M
#   w_hat_ts::TV
#   w_hats::W
#   ergo_q_maps::EM
# end

struct SimResult_NonSOC{T,X,U,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
end

struct SimResult{T,X,U,B,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  bs::B
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
end

struct SimResultWeighted{T,X,U,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
  q_target_maps
end

struct SimResultWeightedSpeed{T,X,U,S,B,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  speeds::S
  bs::B
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
  q_target_maps
end

struct SimResultWeightedSpeedParams{T,X,U,SP,B,S,L,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  speeds::SP
  bs::B
  σ_s::S
  λ_s::L
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
  q_target_maps
end

struct SimResultPre{T,X,M,TV,W,EM}
  ts::T
  xs::X
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
end

function ErgoGrid(ngpkf_grid::G) where {G<:NGPKF.NGPKFGrid}
  origin = (ngpkf_grid.xs[1], ngpkf_grid.ys[1])
  dxs = (Base.step(ngpkf_grid.xs), Base.step(ngpkf_grid.ys))
  Ls = (maximum(ngpkf_grid.xs) - minimum(ngpkf_grid.xs), maximum(ngpkf_grid.ys) - minimum(ngpkf_grid.ys))

  ergo_grid = ErgodicController.Grid(origin, dxs, Ls)

  return ergo_grid
end

# L = dx * (N-1)

function ErgoGrid(ngpkf_grid::G, Ns) where {G<:NGPKF.NGPKFGrid}

  origin = (ngpkf_grid.xs[1], ngpkf_grid.ys[1])
  Ls = (maximum(ngpkf_grid.xs) - minimum(ngpkf_grid.xs), maximum(ngpkf_grid.ys) - minimum(ngpkf_grid.ys))
  dxs = Ls ./ (Ns .- 1)

  # Creates an ergo_grid with origin, spacing which accounts for NGPKF grid and then the plan for direction cosine transform
  ergo_grid = ErgodicController.Grid(origin, dxs, Ls)

  return ergo_grid

end

function ngpkf_to_ergo(ngpkf_grid::G1, ergo_grid::G2, clarity_map) where {G1<:NGPKF.NGPKFGrid,G2<:ErgodicController.Grid}

  Ns = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  if Ns == ergo_grid.N
    return 1.0 * clarity_map
  end

  # interpolate the data
  itp = linear_interpolation((ngpkf_grid.xs, ngpkf_grid.ys), clarity_map, extrapolation_bc=Line())
  ergo_map = itp(ErgodicController.xs(ergo_grid), ErgodicController.ys(ergo_grid))

  return ergo_map

end


function simulate_weighted_exp_SoC_spatial(ts, x0::XS, b0, controllers, soc_profile, w_rated_val;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288; # corresponds to October 15th
  lat = 35.45; # degrees
  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0;
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0 ,soc_profile[1], error_sum, error);
  speeds = [speed];

  # decide the control input for the first step
  u0, q_target = controllers(t0, x0, M, w_rated_val;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax = speed,
    ΔT=ΔT,
  )
  us = [u0,]

  # update ASV state of charge
  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))

  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # Fit new hyperparameters


        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        M = reshape(KF.μ(w_hats[end]), Nx, Ny)

        # Call the real-time speed controller
        speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
        push!(speeds, speed);

        u, q_target = controllers(t, x, M, w_rated_val;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          umax = speed,
          ΔT=ΔT,
        )
        
        push!(q_target_maps, q_target)
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))

    end
  return SimResultWeightedSpeed(ts, xs, us, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)
end


function simulate_weighted_exp_SoC_spatial_param(ts, x0::XS, b0, controllers, soc_profile, w_rated_val;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288; # corresponds to October 15th
  lat = 35.45; # degrees

  # store hyperparameter estimates
  σs = [1.0, ]
  λs = [1.0, ]
  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0;
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0 ,soc_profile[1], error_sum, error);

  # decide the control input for the first step
  u0, q_target = controllers(t0, x0, M, w_rated_val;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax = speed,
    ΔT=ΔT,
  )
  us = [u0,]

  # update ASV state of charge
  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))

  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # Fit new hyperparameters
        σ, λ = Variograms.hp_fit(measurements)
        push!(σs, σ)
        push!(λs, λ)

        # Update KF
        res_factor = 0.4 #l_spatial / sqrt(2.0)

        kern = NGPKF.MaternKernel(σ, 1/λ)

        ngp_grid_x = range(extrema(EnvDataSpatial.X)..., step= res_factor )
        ngp_grid_y = range(extrema(EnvDataSpatial.Y)..., step= res_factor )

        ngpkf_grid = NGPKF.NGPKFGrid(ngp_grid_x, ngp_grid_y, kern)

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        M = reshape(KF.μ(w_hats[end]), Nx, Ny)

        # Call the real-time speed controller
        speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);

        u, q_target = controllers(t, x, M, w_rated_val;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          umax = speed,
          ΔT=ΔT,
        )
        
        push!(q_target_maps, q_target)
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))

    end
  return SimResultWeightedSpeedParams(ts, xs, us, bs, σs, λs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)
end

function simulate_weighted_exp_spatial(ts, x0::XS, controllers, w_rated_val;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # decide the control input for the first step
  u0, q_target = controllers(t0, x0, M, w_rated_val;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    ΔT=ΔT,
  )
  us = [u0,]

  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]


      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        M = reshape(KF.μ(w_hats[end]), Nx, Ny)

        
        u, q_target = controllers(t, x, M, w_rated_val;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          ΔT=ΔT,
        )
        
        push!(q_target_maps, q_target)
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)

    end

  # catch e
    # println(e)
  # end

  return SimResultWeighted(ts, xs, us, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end


function simulate_weighted_spatial(ts, x0::XS, controllers;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # Initiatize the estimate to be equal to the rated value
  # Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  # M = ones(Nx, Ny)
  # M *= -1.5

  # decide the control input for the first step
  u0, q_target = controllers(t0, x0, w_hat;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    ΔT=ΔT,
  )
  us = [u0,]

  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]


      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        # M = reshape(KF.μ(w_hats[end]), Nx, Ny)

        u, q_target = controllers(t, x, w_hats[end];
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          ΔT=ΔT,
        )
        
        push!(q_target_maps, q_target)
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)

    end

  # catch e
    # println(e)
  # end

  return SimResultWeighted(ts, xs, us, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end


function simulate_spatial_cvx_bound(ts, x0::XS, controllers, convex_polygon;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # decide the control input for the first step
  u0 = controllers(t0, x0, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    ΔT=ΔT,
  )
  us = [u0,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]

      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        u = controllers(t, x, convex_polygon;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          ΔT=ΔT,
        )
        
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)

    end

  # catch e
    # println(e)
  # end

  return SimResult_NonSOC(ts, xs, us, measurements, w_hat_ts, w_hats, ergo_q_maps)

end

function simulate_spatial_cvx_bound_speed(ts, x0::XS, b0, controllers, soc_profile, convex_polygon;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288; # corresponds to October 15th
  lat = 35.45; # degrees
  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0;
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0 ,soc_profile[1], error_sum, error);

  speeds = [speed];

  # decide the control input for the first step
  u0 = controllers(t0, x0, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax = speed,
    ΔT=ΔT,
  )
  us = [u0,]

  # update ASV state of charge
  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        # Call the real-time speed controller
        speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
        push!(speeds, speed);

        u = controllers(t, x, convex_polygon;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          umax = speed,
          ΔT=ΔT,
        )
        
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))

    end

  # catch e
    # println(e)
  # end
  q_target_maps = 0;

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end

function simulate_spatial_cvx_bound_speed_param(ts, x0::XS, b0, controllers, soc_profile, convex_polygon;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288; # corresponds to October 15th
  lat = 35.45; # degrees

  # store hyperparameter estimates
  σs = [1.0, ]
  λs = [1.0, ]
  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0;
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0 ,soc_profile[1], error_sum, error);

  speeds = [speed];

  # decide the control input for the first step
  u0 = controllers(t0, x0, convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax = speed,
    ΔT=ΔT,
  )
  us = [u0,]

  # update ASV state of charge
  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # Fit new hyperparameters
        σ, λ = Variograms.hp_fit(measurements)
        push!(σs, σ)
        push!(λs, λ)

        # Update KF
        res_factor = 0.1 #l_spatial / sqrt(2.0)

        kern = NGPKF.MaternKernel(σ, 1/λ)

        ngp_grid_x = range(extrema(EnvDataSpatial.X)..., step= res_factor )
        ngp_grid_y = range(extrema(EnvDataSpatial.Y)..., step= res_factor )

        ngpkf_grid = NGPKF.NGPKFGrid(ngp_grid_x, ngp_grid_y, kern)

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        # Call the real-time speed controller
        speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
        push!(speeds, speed);

        u = controllers(t, x, convex_polygon;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          umax = speed,
          ΔT=ΔT,
        )
        
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))

    end

  # catch e
    # println(e)
  # end
  q_target_maps = 0;

  return SimResultWeightedSpeedParams(ts, xs, us, speeds, bs, σs, λs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end

function simulate_weighted_exp_spatial_cvx_bound(ts, x0::XS, controllers, w_rated_val, convex_polygon;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # decide the control input for the first step
  u0, q_target = controllers(t0, x0, M, w_rated_val,convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    ΔT=ΔT,
  )
  us = [u0,]

  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]


      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        M = reshape(KF.μ(w_hats[end]), Nx, Ny)

        
        u, q_target = controllers(t, x, M, w_rated_val,convex_polygon;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          ΔT=ΔT,
        )
        
        push!(q_target_maps, q_target)
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)

    end

  # catch e
    # println(e)
  # end

  return SimResultWeighted(ts, xs, us, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end

function simulate_weighted_exp_spatial_cvx_bound_speed(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288; # corresponds to October 15th
  lat = 35.45; # degrees
  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0;
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0 ,soc_profile[1], error_sum, error);

  speeds = [speed];

  # decide the control input for the first step
  println("max speed defined: $(speed)")

  u0, q_target = controllers(t0, x0, M, w_rated_val,convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax=0.15,
    ΔT=ΔT,
    P = 0,
  )

  println("Initial control input: $(u0)")
  us = [u0,]

  # update ASV state of charge
  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))

  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        M = reshape(KF.μ(w_hats[end]), Nx, Ny)

        speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
        push!(speeds, speed);

        # println("In Loop")
        # println("speed : $(speed)")
        # println("x : $(x)")
        
        u, q_target = controllers(t, x, M, w_rated_val,convex_polygon;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          umax=0.15,
          ΔT=ΔT,
          P = 1,
        )

        # return q_target
        
        push!(q_target_maps, q_target)
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))

    end

  # catch e
    # println(e)
  # end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end

function simulate_weighted_exp_spatial_cvx_bound_speed_param(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288; # corresponds to October 15th
  lat = 35.45; # degrees

  # store hyperparameter estimates
  σs = [1.0, ]
  λs = [1.0, ]
  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0;
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0 ,soc_profile[1], error_sum, error);

  speeds = [speed];

  # decide the control input for the first step
  u0, q_target = controllers(t0, x0, M, w_rated_val,convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax=speed,
    ΔT=ΔT,
  )
  us = [u0,]

  # update ASV state of charge
  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))

  q_target_maps = [q_target,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # Fit new hyperparameters
        σ, λ = Variograms.hp_fit(measurements)
        push!(σs, σ)
        push!(λs, λ)

        # Update KF
        res_factor = 0.1 #l_spatial / sqrt(2.0)

        kern = NGPKF.MaternKernel(σ, 1/λ)

        ngp_grid_x = range(extrema(EnvDataSpatial.X)..., step= res_factor )
        ngp_grid_y = range(extrema(EnvDataSpatial.Y)..., step= res_factor )

        ngpkf_grid = NGPKF.NGPKFGrid(ngp_grid_x, ngp_grid_y, kern)

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        M = reshape(KF.μ(w_hats[end]), Nx, Ny)

        speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
        push!(speeds, speed);
        
        u, q_target = controllers(t, x, M, w_rated_val,convex_polygon;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          umax=speed,
          ΔT=ΔT,
        )
        
        push!(q_target_maps, q_target)
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))

    end

  # catch e
    # println(e)
  # end

  return SimResultWeightedSpeedParams(ts, xs, us, speeds, bs, σs, λs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end


function simulate_SoC_spatial(ts, x0::XS, controllers, speed_profile;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  
  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # decide the control input for the first step
  u0 = controllers(t0, x0;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax = speed_profile(t0),
    ΔT=ΔT,
  )
  us = [u0,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]


      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        u = controllers(t, x;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          umax = speed_profile(t),
          ΔT=ΔT,
        )
        
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)

    end

  # catch e
    # println(e)
  # end

  return SimResult(ts, xs, us, measurements, w_hat_ts, w_hats, ergo_q_maps)

end


# Simulate spatial domain with SOC-based speed control
function simulate_spatial_w_speed(ts, x0::XS, b0, controllers, soc_profile;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288; # corresponds to October 15th
  lat = 35.45; # degrees

  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0;
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # Compute velocity for first step
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0 ,soc_profile[1], error_sum, error);

  # decide the control input for the first step
  u0 = controllers(t0, x0;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    umax = speed,
    ΔT=ΔT,
  )
  us = [u0,]

  push!(bs, SoCController.batterymodel!(boat, dayOfYear, t0/60, lat, norm(u0), b0, ΔT/60))

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
     
      x = xs[end]
      b = bs[end]


      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
        # speed, error = SoCController.speed_controller(b, soc_profile[it], error)

        traj = vcat(xs...)

        u = controllers(t, x;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          umax = speed,
          ΔT=ΔT,
        )
        
        # Debug 01
        # if isnan(x[1])
        #   println("current state: $(x)")
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if isnan(new_xs[1])
      #   println("New state: $(new_xs)")
      # end
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))

    end

  # catch e
    # println(e)
  # end

  return SimResult(ts, xs, us, bs, measurements, w_hat_ts, w_hats, ergo_q_maps)

end

# Simulate Spatiotemporal 



# Simulate Spatial only 
function simulate_spatial(ts, x0::XS, controllers;
    ngpkf_grid::G,
    EnvDataSpatial,
    σ_meas=0, 
    σ_process=0,
    Q_process = σ_process^2 * I,
    fuse_measurements_every_ΔT=5.0,
    recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}
    
    # extract info from arguments
    t0 = ts[1]
    xs = [x0,]
    N_robots = length(x0)
    ΔT = Base.step(ts)
  
    Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
    ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))
  
    # setup map states
    w_hat_ts = [t0,]
  
    w_hat = NGPKF.initialize(ngpkf_grid)
    w_hats = [w_hat,]

    # check the covariances
    # print(NGPKF.KF.Σ(w_hat))
    # @assert false

    # clarity map
    q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
    ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
    ergo_q_maps = [ergo_q_map,]
  
    # get a measurement
    # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
    measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  
  
    # decide the control input for the first step
    u0 = controllers(t0, x0;
      ngpkf_grid=ngpkf_grid,
      ergo_grid=ergo_grid,
      ergo_q_map=ergo_q_maps[end],
      traj=vcat(xs...),
      ΔT=ΔT,
    )
    us = [u0,]
  
    last_measurement_fuse_time = t0
    last_measurement_fuse_index = 0
  
    last_control_update_time = t0
    # try
      @progress for (it, t) in enumerate(ts[1:(end-1)])
        
        x = xs[end]


        # make a measurement from each robot
        ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
        append!(measurements, ys)
        # check if we need to fuse measurements
        if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT
  
          # # collect all the locations we have made measurements
          # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
          # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)
  
          # # extract x and y components of the measurements
          # measurements_wx = [w[1] for w in measurement_w]
          # measurements_wy = [w[2] for w in measurement_w]
  
          # run NGPKF
          w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

          # grab the data again
          new_measurements = measurements[(last_measurement_fuse_index+1):end]
          measurement_pos = [m.p for m in new_measurements]
          measurements_w = [m.y for m in new_measurements]

          # run the fusion
          w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
          
          # save the new maps
          push!(w_hat_ts, t)
          push!(w_hats, w_hat_new)
  
          # update the clarity map
          q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
          ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
          push!(ergo_q_maps, ergo_q_map)
  
          last_measurement_fuse_time = t
          last_measurement_fuse_index = length(measurements)
        end
  
  
        # if (t - last_control_update_time >= recompute_controller_every_ΔT)
        #   # chose a control action
  
          traj = vcat(xs...)

          u = controllers(t, x;
            ngpkf_grid=ngpkf_grid,
            ergo_grid=ergo_grid,
            ergo_q_map=ergo_q_maps[end], # current clarity
            traj=traj,  # list of all points visited by all agents
            ΔT=ΔT,
          )
          
          # Debug 01
          # if isnan(x[1])
          #   println("current state: $(x)")
          # end
  

          push!(us, u)
  
        #   last_control_update_time = t
  
        # end

        # update 
        u = us[end] # use the last control input
        new_xs = step(t, xs[end], u, ΔT)

        push!(bs, SoCController.batterymodel!())

        # if isnan(new_xs[1])
        #   println("New state: $(new_xs)")
        # end
        push!(xs, new_xs)
  
      end
  
    # catch e
      # println(e)
    # end
  
    return SimResult(ts, xs, us, measurements, w_hat_ts, w_hats, ergo_q_maps)
  
  end
  
# Simulate Spatiotemporal 

# Simulate Spatial only 
function simulate_prerec_traj(ts, xs_rec;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  
  # extract info from arguments
  t0 = ts[1]
  x0 = xs_rec[1] # Get the start state from the pre-recorded trajectory

  xs = [x0,]
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  w_hat = NGPKF.initialize(ngpkf_grid)
  w_hats = [w_hat,]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  


  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0
  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])
      
      x = xs[end]


      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)
      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # run NGPKF
        w_hat = NGPKF.predict(ngpkf_grid, w_hats[end]; Q_process=Q_process)

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_w = [m.y for m in new_measurements]

        # run the fusion
        w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end
      # update 
      curr_traj_length = size(xs,1)
      new_xs = xs_rec[curr_traj_length + 1]

      push!(xs, new_xs)

      # us = @SVector[0, 0.]

    end

  # catch e
    # println(e)
  # end

  return SimResultPre(ts, xs, measurements, w_hat_ts, w_hats, ergo_q_maps)

end

# Simulate Spatiotemporal 

function simulate(ts, x0::XS, controllers;
  ngpkf_grid::G,
  EnvDataSpatial,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  N_robots = length(x0)
  ΔT = Base.step(ts)
  
  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  wx_hat = NGPKF.initialize(ngpkf_grid)
  wx_hats = [wx_hat,]

  wy_hat = NGPKF.initialize(ngpkf_grid)
  wy_hats = [wy_hat,]

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, wx_hat, wy_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  
  # println(q_map)
  # println("size of: " $(size(ergo_q_map)))

  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataSpatial; σ_meas = σ_meas)...]  

  # decide the control input for the first step
  u0 = controllers(t0, x0;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    ΔT=ΔT,
  )

  us = [u0,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0

  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])

      x = xs[end]
 
      # make a measurement from each robot
      ys = measure(t, x, EnvDataSpatial; σ_meas=σ_meas)
      append!(measurements, ys)

      

      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        wx_hat = NGPKF.predict(ngpkf_grid, wx_hats[end]; Q_process=Q_process)
        wy_hat = NGPKF.predict(ngpkf_grid, wy_hats[end]; Q_process=Q_process)
        

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_wx = [m.y[1] for m in new_measurements]
        measurements_wy = [m.y[2] for m in new_measurements]

        # run the fusion
        wx_hat_new = NGPKF.correct(ngpkf_grid, wx_hat, measurement_pos, measurements_wx; σ_meas=σ_meas)
        wy_hat_new = NGPKF.correct(ngpkf_grid, wy_hat, measurement_pos, measurements_wy; σ_meas=σ_meas)

        # save the new maps
        push!(w_hat_ts, t)
        push!(wx_hats, wx_hat_new)
        push!(wy_hats, wy_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, wx_hat_new, wy_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        u = controllers(t, x;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          ΔT=ΔT,
        )
        
        push!(us, u)

      #   last_control_update_time = t

      # end



      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)
      push!(xs, new_xs)

    end

  # catch e
    # println(e)
  # end

  return SimResult(ts, xs, us, measurements, w_hat_ts, wx_hats, wy_hats, ergo_q_maps)

end

end