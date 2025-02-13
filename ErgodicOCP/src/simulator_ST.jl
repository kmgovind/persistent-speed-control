module SimulatorST

using LinearAlgebra, StaticArrays, Interpolations
using ProgressLogging
using SpatiotemporalGPs
using ..SyntheticData, ..NGPKF, ..ErgodicController, ..KF, ..SoCController, ..Variograms, ..JordanLakeDomain


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

function measure(t, p::SV, data::EDST; σ_meas=0, Q_meas=σ_meas * I) where {EDST<:EnvDataST,SV<:SVector{2}}
  # y = data(p..., t) + (σ_meas * randn(1))[1]
  y = data(p..., t) + (0.0 * randn(1))[1]
  return MeasurementSpatial(t, p, y)
end

function measure(t, ps::VSV, data::EDST; σ_meas=0, Q_meas = σ_meas * I) where {EDST<:EnvDataST,SV<:SVector{2}, VSV <: AbstractVector{SV}}
  return [measure(t, p, data; Q_meas=Q_meas) for p in ps]

end

# function measure(data, x, y, t, σ_m=0.1)
#   return data.itp(x, y, t) + σ_m *randn()
# end

function measure(data, x, y, t, σ_m=0.0)
  return data.itp(x, y, t) + σ_m *randn()
end

function measure_reconstruction(data, t_idx)
  return data[t_idx]
end


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

"""
Function to take the STGPKF format measurements into 
"""
function data2struct(old_measurements, pos, ws, ts)

  for idx = 1:length(ts)
    push!(old_measurements, MeasurementSpatial(ts[idx], pos[idx], ws[idx]))
  end

  return old_measurements
end


struct SimResult{T,X,U,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
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

struct SimResultWeightedSpeedParams{T,X,U,SP,B,S,LS,LT,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  speeds::SP
  bs::B
  σ_s::S
  λx_s::LS
  λt_s::LT
  measurements::M
  w_hat_ts::TV
  w_hats::W
  ergo_q_maps::EM
  q_target_maps
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

# Simulate pre-recorded trajectory 


# Simulate Spatial only 
function simulate_spatial(ts, x0::XS, controllers;
    ngpkf_grid::G,
    EnvDataST,
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
    # ys = [measure(t0, x0[i], EnvDataST; σ_meas=σ_meas) for i = 1:N_robots]
    println("going into measurements")
    measurements = [measure(t0, x0, EnvDataST; σ_meas = σ_meas)...]  
    println("going out of measurements")
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
        ys = measure(t, x, EnvDataST; σ_meas=σ_meas)
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
          
        #   println("u : $(u)")
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
  
    return SimResult(ts, xs, us, measurements, w_hat_ts, w_hats, ergo_q_maps)
  
  end
  
# Simulate Spatiotemporal 
function simulate(ts, x0::XS, controllers;
  ngpkf_grid::G,
  EnvDataST,
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
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataST; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, EnvDataST; σ_meas = σ_meas)...]  

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
      ys = measure(t, x, EnvDataST; σ_meas=σ_meas)
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

# Spatiotemporal Jordan Lake Simulation with known hyperparameters
function simulate_known_param(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
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

  Nx, Ny= length(ngpkf_grid.xs), length(ngpkf_grid.ys) 
  ergo_grid = ErgoGrid(ngpkf_grid, (length(EnvData.xs), length(EnvData.ys)))


  # setup map states
  w_hat_ts = [t0,]
  stg_hat_ts = [t0, ]
  

  # w_hat = NGPKF.initialize(ngpkf_grid)
  state = stgpkf_initialize(stgp_problem)
  # states = (typeof(state))[]
  # w_hat = state
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  # w_hat = reshape(estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1], Nx, Ny)
  w_hats = [w_hat, ]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  # q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # measure_σ = 0.1 # m/s
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  # measurements = [measure(t0, x0, EnvData; σ_meas = σ_meas)...]  
  # measurements = [measure(EnvData, x0[1]..., EnvData.ts[1], σ_meas)...]

  LookUpData = EnvData.data[1,1,:]
  # println("EnvData: ", EnvData)
  measurements = [measure_reconstruction(LookUpData, 1)...]

  # measure_Σ = (measure_σ^2) * I(10);

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0, soc_profile[1], error_sum, error);

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

      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        measurement_pos = vec([@SVector[xs[i][1][1],xs[i][1][2]] for i in last_measurement_fuse_index+1:length(xs)])
        measurements_w = measurements[(last_measurement_fuse_index+1):end]
        measure_Σ = (σ_meas^2) * I(length(measurements_w));


        # run the fusion
        # do the KF correction
        state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
        # push!(states, state_correction)
        state = stgpkf_predict(stgp_problem, state_correction)
        # w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # STGPKF.update_and_predict!(estimator_1, key_parameters_1, yᵢ, 𝐬ᵢᵗⁱˡᵈᵉ)
        
        est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
        w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        # q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        # ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
        q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      # ys = measure(t, x, EnvData; σ_meas=σ_meas)
      # ys =  [measure(EnvData, x..., ts_hrs*60, σ_meas)...]
      # ys = [measure(EnvData, x[end]..., EnvData.ts[1], σ_meas)...]
      # ys = [measure(EnvData, x[end]..., EnvData.ts[it], σ_meas)...]
      ys = [measure_reconstruction(LookUpData, it)...]
      append!(measurements, ys)

      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        # M = reshape(KF.μ(w_hats[end]), Nx, Ny)
        M = w_hats[end]

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

        # if any(isnan.(u[1]))
        #   println("u: ", u[end])
        #   println("New state: $(xs[end])")
        #   u[end] = [sign(0.7 - xs[end][1][1])*0.1,sign(3.25 - xs[end][1][2])*0.1]
        # end


        push!(us, u)


      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

    
      # if any(isnan.(new_xs[1]))
      #   println("New state: $(new_xs)")
      #   println("u: ", u)
      #   println("speed: ", speed)
      # end
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))
    end

  # catch e
    # println(e)
  # end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end

# Spatiotemporal Jordan Lake Simulation with estimated parameters
function simulate_param_est(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  boat = SoCController.ASV_Params();
  dayOfYear = 288; # corresponds to October 15th
  lat = 35.45; # degrees

  # store hyperparameter estimates
  σs = [1.0, ]
  λxs = [1.0, ]
  λts = [1.0, ]

  measure_vec = []

  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  bs = ones(1)*b0;
  N_robots = length(x0)
  ΔT = Base.step(ts)

  Nx, Ny= length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  # w_hat = NGPKF.initialize(ngpkf_grid)
  state = stgpkf_initialize(stgp_problem)
  # states = (typeof(state))[]
  # w_hat = state
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  # w_hat = reshape(estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1], Nx, Ny)
  w_hats = [w_hat, ]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  # q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  # ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  # ergo_q_maps = [ergo_q_map,]

  qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]


  # get a measurement
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(EnvData, x0[1]..., EnvData.ts[1], σ_meas)...]

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

      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        measurement_pos = vec([@SVector[xs[i][1][1],xs[i][1][2]] for i in last_measurement_fuse_index+1:length(xs)])
        measurements_w = measurements[(last_measurement_fuse_index+1):end]
        measure_Σ = (σ_meas^2) * I(length(measurements_w));

        measure_vec = data2struct(measure_vec, measurement_pos, measurements_w, ts[it-length(measurements_w)+1:it])

        # setup the spatial and temporal kernels
        σt = 2.0   # m/s
        # σs = 1.0   # m/s
        # lt = 0.25*60.0  # minutes
        # ls = 0.75   # km

        if length(measure_vec) > 1000
          σ, λx, λt = Variograms.hp_fit(measure_vec[end-1000:end])
        else
          σ, λx, λt = Variograms.hp_fit(measure_vec)
        end


        # σ, λx, λt = Variograms.hp_fit(measure_vec)
        σ = max(σ, 0.001)
        λx = max(λx, 0.001)
        λt = max(λt, 0.001)
        push!(σs, σ)
        push!(λxs, λx)
        push!(λts, λt)

        kt = Matern(1/2, σt, λt)
        ks = Matern(1/2, σ, λx)

        # determine the spatial step size
        Δx = 0.10 # km

        # create the spatial domain
        gridxs = 0:Δx:1.4
        gridys = 0:Δx:6.5

        grid_pts = vec([@SVector[x, y] for x in gridxs, y in gridys]);

        stgp_problem = STGPKFProblem(grid_pts, ks, kt, ΔT)
        kern = ks
        ngpkf_grid = NGPKF.NGPKFGrid(EnvData.xs, EnvData.ys, kern)
        ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))       


        # run the fusion
        # do the KF correction
        state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
        # push!(states, state_correction)
        state = stgpkf_predict(stgp_problem, state_correction)
        # w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # STGPKF.update_and_predict!(estimator_1, key_parameters_1, yᵢ, 𝐬ᵢᵗⁱˡᵈᵉ)
        
        est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
        w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        # q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        # ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
        q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end

      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      # ys = measure(t, x, EnvData; σ_meas=σ_meas)
      # ys =  [measure(EnvData, x..., ts_hrs*60, σ_meas)...]

      # ys = [measure(EnvData, x[end]..., EnvData.ts[1], σ_meas)...]
      ys = [measure(EnvData, x[end]..., EnvData.ts[it], σ_meas)...]
      append!(measurements, ys)


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        # M = reshape(KF.μ(w_hats[end]), Nx, Ny)
        M = w_hats[end]

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
        println("u: ", u[1])
        if isnan.(u[1][1]) || isnan.(u[1][2])
          u = [@SVector[0.0,0.0]]
        end
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
  
  return SimResultWeightedSpeedParams(ts, xs, us, speeds, bs, σs, λxs, λts, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end

# Spatiotemporal Jordan Lake Simulation with known hyperparameters
function simulate_reconstruction(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
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

  Nx, Ny= length(ngpkf_grid.xs), length(ngpkf_grid.ys) # The grid of size (64, 32)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]
  stg_hat_ts = [t0, ]
  

  # w_hat = NGPKF.initialize(ngpkf_grid)
  state = stgpkf_initialize(stgp_problem)
  # states = (typeof(state))[]
  # w_hat = state
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  # w_hat = reshape(estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1], Nx, Ny)
  w_hats = [w_hat, ]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  # q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # measure_σ = 0.1 # m/s
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  # measurements = [measure(t0, x0, EnvData; σ_meas = σ_meas)...]  
  measurements = [measure(EnvData, x0[1]..., EnvData.ts[1], σ_meas)...]
  # measure_Σ = (measure_σ^2) * I(10);

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0, soc_profile[1], error_sum, error);

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

      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        measurement_pos = vec([@SVector[xs[i][1][1],xs[i][1][2]] for i in last_measurement_fuse_index+1:length(xs)])
        measurements_w = measurements[(last_measurement_fuse_index+1):end]
        measure_Σ = (σ_meas^2) * I(length(measurements_w));


        # run the fusion
        # do the KF correction
        state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
        # push!(states, state_correction)
        state = stgpkf_predict(stgp_problem, state_correction)
        # w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # STGPKF.update_and_predict!(estimator_1, key_parameters_1, yᵢ, 𝐬ᵢᵗⁱˡᵈᵉ)
        
        est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
        w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        # q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        # ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
        q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      # ys = measure(t, x, EnvData; σ_meas=σ_meas)
      # ys =  [measure(EnvData, x..., ts_hrs*60, σ_meas)...]
      # ys = [measure(EnvData, x[end]..., EnvData.ts[1], σ_meas)...]
      ys = [measure(EnvData, x[end]..., EnvData.ts[it], σ_meas)...]
      append!(measurements, ys)


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        # M = reshape(KF.μ(w_hats[end]), Nx, Ny)
        M = w_hats[end]

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

        # if any(isnan.(u[1]))
        #   println("u: ", u[end])
        #   println("New state: $(xs[end])")
        #   u[end] = [sign(0.7 - xs[end][1][1])*0.1,sign(3.25 - xs[end][1][2])*0.1]
        # end


        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

    
      # if any(isnan.(new_xs[1]))
      #   println("New state: $(new_xs)")
      #   println("u: ", u)
      #   println("speed: ", speed)
      # end
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))
    end

  # catch e
    # println(e)
  # end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end

function simulate_transect(ts, x0::XS, b0, controllers, soc_profile, w_rated_val, convex_polygon, stgp_problem;
  ngpkf_grid::G,
  EnvData,
  transect_pts,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0/60,
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

  Nx, Ny= length(ngpkf_grid.xs), length(ngpkf_grid.ys) 
  ergo_grid = ErgoGrid(ngpkf_grid, (length(EnvData.xs), length(EnvData.ys)))

  # setup map states
  w_hat_ts = [t0,]
  stg_hat_ts = [t0, ]
  

  # w_hat = NGPKF.initialize(ngpkf_grid)
  state = stgpkf_initialize(stgp_problem)
  # states = (typeof(state))[]
  # w_hat = state
  est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
  w_hat = reshape(est, length(EnvData.xs), length(EnvData.ys))
  # w_hat = reshape(estimator.𝐱ʰᵃᵗⱼₗᵢ[:, 1], Nx, Ny)
  w_hats = [w_hat, ]

  # check the covariances
  # print(NGPKF.KF.Σ(w_hat))
  # @assert false

  # clarity map
  # q_map = NGPKF.clarity_map(ngpkf_grid, w_hat)
  qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
  q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # measure_σ = 0.1 # m/s
  # ys = [measure(t0, x0[i], EnvDataSpatial; σ_meas=σ_meas) for i = 1:N_robots]
  # measurements = [measure(t0, x0, EnvData; σ_meas = σ_meas)...]  
  # measurements = [measure(EnvData, x0[1]..., EnvData.ts[1], σ_meas)...]
  # measure_Σ = (measure_σ^2) * I(10);

  LookUpData = EnvData.data[1,1,:]
  measurements = [measure_reconstruction(LookUpData, 1)...]

  # Initiatize the estimate to be equal to the rated value
  Nx, Ny = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  M = ones(Nx, Ny)
  M *= w_rated_val

  # Call the real-time speed controller
  error = 0.0;
  error_sum = 0.0;
  speed, error_sum, error = SoCController.speed_controller(b0, soc_profile[1], error_sum, error);

  speeds = [speed];

  # Initialize waypoint index
  waypoint_idx = 1

  # decide the control input for the first step
  u0, q_target, waypoint_idx = controllers(t0, x0, M, w_rated_val,convex_polygon;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    transect_pts=transect_pts,
    waypoint_idx=waypoint_idx,
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

      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        measurement_pos = vec([@SVector[xs[i][1][1],xs[i][1][2]] for i in last_measurement_fuse_index+1:length(xs)])
        measurements_w = measurements[(last_measurement_fuse_index+1):end]
        measure_Σ = (σ_meas^2) * I(length(measurements_w));


        # run the fusion
        # do the KF correction
        state_correction = stgpkf_correct(stgp_problem, state, measurement_pos, measurements_w, measure_Σ)
        # push!(states, state_correction)
        state = stgpkf_predict(stgp_problem, state_correction)
        # w_hat_new = NGPKF.correct(ngpkf_grid, w_hat, measurement_pos, measurements_w; σ_meas=σ_meas)
        
        # STGPKF.update_and_predict!(estimator_1, key_parameters_1, yᵢ, 𝐬ᵢᵗⁱˡᵈᵉ)
        
        est = SpatiotemporalGPs.STGPKF.get_estimate(stgp_problem, state)
        w_hat_new = reshape(est, length(EnvData.xs), length(EnvData.ys))

        # save the new maps
        push!(w_hat_ts, t)
        push!(w_hats, w_hat_new)

        # update the clarity map
        # q_map = NGPKF.clarity_map(ngpkf_grid, w_hat_new)
        # ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        qs = SpatiotemporalGPs.STGPKF.get_estimate_clarity(stgp_problem, state)
        q_map = reshape(qs, length(EnvData.xs), length(EnvData.ys))
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end
      
      x = xs[end]
      b = bs[end]

      # make a measurement from each robot
      # ys = measure(t, x, EnvData; σ_meas=σ_meas)
      # ys =  [measure(EnvData, x..., ts_hrs*60, σ_meas)...]
      # ys = [measure(EnvData, x[end]..., EnvData.ts[1], σ_meas)...]
      # ys = [measure(EnvData, x[end]..., EnvData.ts[it], σ_meas)...]
      # append!(measurements, ys)

      ys = [measure_reconstruction(LookUpData, it)...]
      append!(measurements, ys)


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        # M = reshape(KF.μ(w_hats[end]), Nx, Ny)
        M = w_hats[end]

        speed, error_sum, error = SoCController.speed_controller(b, soc_profile[it], error_sum, error);
        push!(speeds, speed);
        
        u, q_target, waypoint_idx = controllers(t, x, M, w_rated_val,convex_polygon;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          transect_pts=transect_pts,
          waypoint_idx=waypoint_idx,
          umax=speed,
          ΔT=ΔT,
        )
        
        push!(q_target_maps, q_target)
      

        push!(us, u)

      #   last_control_update_time = t

      # end

      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)

      # if any(isnan.(new_xs[1]))
      #   println("New state: $(new_xs)")
      #   println("u: ", u)
      #   println("speed: ", speed)
      # end
      push!(xs, new_xs)
      push!(bs, SoCController.batterymodel!(boat, dayOfYear, t/60, lat, norm(u), b, ΔT/60))
    end

  # catch e
    # println(e)
  # end

  return SimResultWeightedSpeed(ts, xs, us, speeds, bs, measurements, w_hat_ts, w_hats, ergo_q_maps, q_target_maps)

end

end