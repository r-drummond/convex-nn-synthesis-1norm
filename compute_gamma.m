function [gamma_val,problem_sol,Y_val , Y0_val,Tz_val, Tg_val, W_val,Wu_val, Wf_val ,Wfu_val ] = compute_gamma(W,Wu,Wf,Wfu,bound_u,tol_eps)

n = max(size(W));
nx = size(Wu,2);

% n_mpc = size(G,2);
% W = D;
% Wu = S; b = w;
% Wfu = -H\F;  Wf = -H\G';
% F_store = F;
% Wfu = [1, zeros(1,n/2-1)]*Wfu;Wf = [1, zeros(1,n/2-1)]*Wf;
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
Yf = Tg*Wf
% Y = sdpvar(n,n);
% Y0 = sdpvar(n,nu);
% Yfu = sdpvar(ny,nu);
% Yf = sdpvar(ny,n);
Yf2 = [Yf;-Yf]; Yfu2 = [Yfu;-Yfu];

tau_u = sdpvar;
tau_u2 = sdpvar;

% gamma_sq = sdpvar(nu,nu,'diag'); gamma_sq_abs = sdpvar(nu,nu,'diag');

ones_vec_u = ones(2*nu,1); ones_vec_g = ones(2*ny,1);
gamma_u = sdpvar;
gamma_u = 0;
gamma = sdpvar;

%%
performance = blkdiag(zeros(N-1,N-1),-1*gamma);
performance(1:2*ny,N) = ones_vec_g;
% performance(2*ny+1:2*ny+2*nu,N) = -gamma_u*ones_vec_u;
performance = performance+performance';

% performance(2*ny+1:2*ny+2*nu,2*ny+1:2*ny+2*nu) = -blkdiag(gamma_sq_abs,gamma_sq_abs);
% performance(N-1-nu+1:N-1,N-1-nu+1:N-1) = -gamma_sq;

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
% F = [F,tau_u >= eps*eye(nu)]; F = [F,tau_u2>= eps*eye(2*nu)];
F = [F,tau_u >= eps]; F = [F,tau_u2>= eps];
eps =1*1e-8;
F = [F,Tz >= eps*eye(n)];
% eps =1*1e-12;
F = [F,gamma >= eps];
% F = [F,gamma == eps];
% F = [F,gamma_u >= eps];
% F = [F,gamma_sq >= eps*eye(nu)];
% F = [F,gamma_sq_abs >= eps*eye(nu)];

Mat2 = performance+bounds+robustness;

%%
n_mat = max(size(Mat2));
eps =1*1e-8;
F = [F, Mat2 <= -eps*eye(n_mat)];

%% Impose the constraints on the nn weights and biases;
% tol_eps = 1e-2;
% F = [F,reshape(Y0-Tz*Wu,nu*n,1)<= reshape(Tz*tol_eps*ones(n,nu),nu*n,1)];
% F = [F,reshape(Y0-Tz*Wu,nu*n,1)>= -reshape(Tz*tol_eps*ones(n,nu),nu*n,1)];
% 
% F = [F,reshape(Yf-Tg*Wf,1,n*ny)<= reshape(Tg*ones(ny,n)*tol_eps,1,ny*n)]; F = [F,reshape(Yf-Tg*Wf,1,n*ny)>= -reshape(Tg*ones(ny,n)*tol_eps,1,ny*n)];
% F = [F,reshape(Yfu-Tg*Wfu,1,nx*ny)<=reshape(Tg*ones(ny,nu)*tol_eps,1,ny*nu)]; F = [F,reshape(Yfu-Tg*Wfu,1,nx*ny)>= -reshape(Tg*ones(ny,nu)*tol_eps,1,ny*nu)];
% 
% F = [F,reshape(Y-Tz*W,n^2,1)<=reshape(Tz*ones(n,n)*tol_eps,n^2,1)];
% F = [F,reshape(Y-Tz*W,n^2,1)>=-reshape(Tz*ones(n,n)*tol_eps,n^2,1)];

%%
% obj = gamma+gamma_u;
obj= gamma;
% obj= gamma_u;
% obj = gamma + trace(gamma_sq);
% obj = gamma+gamma_u+trace(gamma_sq+gamma_sq_abs);
% obj = gamma+trace(gamma_sq+gamma_sq_abs);
% obj = trace(gamma_sq)+ gamma+gamma_u;
% obj = [];
% opt_details.set_solver = 'mosek';
opt_details.set_solver = 'sedumi';

sol_2 = solvesdp(F,obj,sdpsettings('solver',opt_details.set_solver));

gamma_val = value(gamma);
% gamma_val = value(gamma_u);
% gamma_u_val = value(gamma_u);
% gamma_u_sq_val = trace(value(gamma_sq));

% gammas = [gamma_val,gamma_u_val,gamma_u_sq_val]
% gammas = [gamma_val,gamma_u_val]

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
























