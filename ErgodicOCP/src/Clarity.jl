module Clarity

    using KernelFunctions, ForwardDiff

    import ..KF, ..NGPKF

    export dt, xd, ys, ngpkf_grid

    ## Variables
    # Define a Matern Kernel
    σ_spatial = 2.0;
    l_spatial = 1.0;
    res_factor = 0.4; # l_spatial/sqrt(2.0)
    kern = NGPKF.MaternKernel(σ_spatial, l_spatial);

    # Define clarity map domain
    dt = 5.0/60.0; # seconds
    xs = 0:0.1:10;
    ys = 0:0.1:10;
    ngpkf_grid = NGPKF.NGPKFGrid(xs, ys, kern);


    ## Methods
    function clarity_prediction(t, q0, C, R, Q)

        k = C / sqrt(Q * R)
        
        q∞ = k / (1 + k)
    
        γ1 = q∞ - q0
        γ2 = γ1 * (k-1)
        γ3 = (k-1) * q0 - k
    
        return q∞ * ( 1 + 2 * γ1 / (γ2 + γ3 * exp(2 * k * Q * t)))
    end
    
        
    function clarity_time(q0, qf, C, R, Q; tmax=10.0)
        
        println("q0: $(q0)")
        println("qf: $(qf)")
        println("C: $(C)")
        println("R: $(R)")
        println("Q: $(Q)")
        
        if q0 >= qf
            return 0.0
        end
    
        k = C / sqrt(Q * R)
        println("k: $(k)")
        
        q∞ = k / (1 + k)
        println("q∞: $(q∞)")
        γ1 = q∞ - q0
        γ2 = γ1 * (k-1)
        γ3 = (k-1) * q0 - k
    
        
            
        if qf >= q∞
            return tmax
        end
    
        t = log((2*q∞*γ1 - qf*γ2 + q∞*γ2)/((qf - q∞)*γ3))/(2*k*Q)
        println("t: $(t)")
    
        return min(t, tmax)
    end
    

    function Cfun(p, x)
        return kern(x, p)^2 / kern(p, p)
    end

    function Rfun(p, x)
        return (kern(x,x) - kern(x, p)^2 / kern(p, p) + 0.5^2)/(dt)
    end

    C_ = Cfun(0,0)
    R_ = Rfun(0,0)

    k = (C_^2 / R_)

    S(p, x)  = Cfun(p, x)^2 / Rfun(p, x)
    DxS(p, x) = ForwardDiff.gradient(xx-> S(p, xx), x)

    function Clarity_delta_t(current_clarity, target_clarity)
        delta_t = (target_clarity - current_clarity) / ((target_clarity - 1) * k * (current_clarity - 1))
        return delta_t
    end

    function Clarity_delta_new(current_clarity, target_clarity)
        den = -target_clarity*k + k*current_clarity*target_clarity + k - k*current_clarity
        return delta_t = (target_clarity - current_clarity) / den
    end

end # module Clarity