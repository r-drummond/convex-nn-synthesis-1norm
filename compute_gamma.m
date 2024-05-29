function [gamma_val,problem_sol,Y_val , Y0_val,Tz_val, Tg_val, W_val,Wu_val, Wf_val ,Wfu_val ] = compute_gamma(W,Wu,Wf,Wfu,bound_u,tol_eps)

% This function  solves the SDP to compute the robustness bound.

%%
n = max(size(W));
ny = size(Wfu,1);
nu = 2;
N = 2*ny+2*nu+n+nu+1;

Tg = sdpvar(ny,ny,'diag');
Tg2 = blkdiag(Tg,Tg);
Tu = sdpvar(nu,nu,'diag');
Tu_mod = sdpvar(nu,nu,'diag');
Tu2 = blkdiag(Tu,Tu_mod);
Tz = sdpvar(n,n,'diag');

Y = Tz*W;
Y0 = Tz*Wu;
Yfu = Tg*Wfu;
Yf = Tg*Wf;

Yf2 = [Yf;-Yf]; Yfu2 = [Yfu;-Yfu];

tau_u = sdpvar;
tau_u2 = sdpvar;

ones_vec_g = ones(2*ny,1);
gamma = sdpvar;

%%
performance = blkdiag(zeros(N-1,N-1),-1*gamma);
performance(1:2*ny,N) = ones_vec_g;
performance = performance+performance';

%%
bounds = blkdiag(zeros(N-1-nu,N-1-nu),-tau_u*eye(nu),(tau_u+tau_u2)*bound_u^2);
bounds(2*ny+1:2*ny+2*nu,2*ny+1:2*ny+2*nu) = -tau_u2*eye(2*nu);

%%
robustness = blkdiag(-0.5*Tg2,zeros(N-2*ny,N-2*ny));

robustness(1:2*ny,2*ny+2*nu+1:2*ny+2*nu+n) = Yf2;
robustness(1:2*ny,2*ny+2*nu+n+1:2*ny+2*nu+n+nu) = Yfu2;

robustness(2*ny+1:2*ny+2*nu,2*ny+1:2*ny+2*nu)=-0.5*Tu2;
robustness(2*ny+1:2*ny+2*nu,2*ny+2*nu+n+1:2*ny+2*nu+n+nu) = [Tu;-Tu_mod];

robustness(2*ny+2*nu+1:2*ny+2*nu+n,2*ny+2*nu+1:2*ny+2*nu+n) = 0.5*(Y-Tz);
robustness(2*ny+2*nu+1:2*ny+2*nu+n,2*ny+2*nu+n+1:2*ny+2*nu+n+nu) = Y0;

robustness = robustness + robustness';

%%
eps = 1*1e-8;
F = [];
F = [F,Tg >= eps*eye(ny)];
F = [F,Tu >= eps*eye(nu)];
F = [F,Tu_mod >= eps*eye(nu)];
F = [F,tau_u >= eps]; F = [F,tau_u2>= eps];
eps =1*1e-8;
F = [F,Tz >= eps*eye(n)];
% eps =1*1e-12;
F = [F,gamma >= eps];

Mat2 = performance+bounds+robustness;

%%
n_mat = max(size(Mat2));
eps =1*1e-8;
F = [F, Mat2 <= -eps*eye(n_mat)];
obj= gamma;
opt_details.set_solver = 'mosek';
% opt_details.set_solver = 'sedumi';

sol_2 = solvesdp(F,obj,sdpsettings('solver',opt_details.set_solver));

gamma_val = value(gamma);
problem_sol = sol_2.problem

%%
Y_val = value(Y);
Y0_val = value(Y0);
Tz_val = value(Tz);
Tg_val = value(Tg);

W_val = Tz_val\Y_val;
Wu_val = Tz_val\Y0_val;
Wf_val = Tg_val\value(Yf);
Wfu_val = Tg_val\value(Yfu);


comp_W = [W_val,W];
comp_Wf = [Wf_val; Wf];
comp_Wfu = [Wfu_val; Wfu];
comp_Wu = [Wu_val; Wu];

norm_W = norm(W_val-W); norm_Wf = norm(Wf_val- Wf); norm_Wfu = norm(Wfu_val- Wfu); norm_Wu = norm(Wu_val- Wu);
norms = [norm_W,norm_Wf,norm_Wfu,norm_Wu]
end
























