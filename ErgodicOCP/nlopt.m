clc; clear; close all;

% Define grid
xs = 0:0.5:10;
ys = 0:0.5:10;
dim_x = length(xs);
dim_y = length(ys);

% Create clarity map and target clarity map
clarity_map = zeros(dim_x, dim_y);
target_clarity = zeros(dim_x, dim_y);

% Define clarity dynamics parameters
Q = 0.01;  % Process noise (decay rate)
R = 0.1;   % Measurement noise
C = 1.0;   % Measurement strength
sensing_radius = 2.5;  % Radius of the sensing area
alpha = 0.1;  % Energy dissipation rate
umax = 2.35;  % Maximum control input
T = 20;  % Reduced time horizon for debugging

% Define target clarity peaks
peaks = [2.5, 2.5; 2.5, 7.5; 7.5, 2.5; 7.5, 7.5];
for p = 1:size(peaks, 1)
    px = peaks(p, 1);
    py = peaks(p, 2);
    for i = 1:dim_x
        for j = 1:dim_y
            target_clarity(i, j) = target_clarity(i, j) + 0.9 * exp(-((xs(i) - px)^2 + (ys(j) - py)^2) / 4);
        end
    end
end

% Decision variables: [x, y, u_x, u_y, b]
x0 = [repmat([5; 2], T, 1); zeros(2*T, 1); 100*ones(T, 1)];

% Bounds
lb = [-inf*ones(2*T,1); -umax*ones(2*T,1); zeros(T,1)];
ub = [inf*ones(2*T,1); umax*ones(2*T,1); inf*ones(T,1)];

% Constraints function
nonlcon = @(x) constraints(x, T, alpha, R, umax);

% Objective function
obj_fun = @(x) objective(x, T, xs, ys, dim_x, dim_y, target_clarity, C, R, Q, sensing_radius);

% Solve using fmincon
options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'iter', 'MaxIterations', 1000);
[x_opt, fval, exitflag] = fmincon(obj_fun, x0, [], [], [], [], lb, ub, nonlcon, options);

% Display results
if exitflag > 0
    disp('Optimal solution found.');
else
    disp('Solver did not find an optimal solution.');
end

% Extract trajectory
x_traj = x_opt(1:T);
y_traj = x_opt(T+1:2*T);
u_x_traj = x_opt(2*T+1:3*T);
u_y_traj = x_opt(3*T+1:4*T);
b_traj = x_opt(4*T+1:5*T);

% Save results
save('single_ocp_results.mat', 'x_traj', 'y_traj', 'u_x_traj', 'u_y_traj', 'b_traj', 'target_clarity');

% Gaussian function for clarity update
function w = gaussian(x, y, cx, cy, sigma)
    w = exp(-((x - cx)^2 + (y - cy)^2) / (2 * sigma^2));
end
