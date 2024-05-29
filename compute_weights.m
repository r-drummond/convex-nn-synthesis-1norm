function [gamma_val,problem_sol,Y_val , Y0_val,Tz_val, Tg_val, W_val,Wu_val, Wf_val ,Wfu_val ] = compute_weights(D,S,w,N,nx,H,F,G,A,B,bound_u,tol_eps)
% This function contains the SDP for solving the constrained optimisation
% over the weights and biases of the NN to improve its robustness. 

%%
n = max(size(D));
nx = size(S,2);

W = D;
Wu = S; 
Wfu = -H\F;  Wf = -H\G';
ny = size(Wfu,1);
nu = 2;

N = 2*ny+2*nu+n+nu+1;

Tg = sdpvar(ny,ny,'diag');
Tg2 = blkdiag(Tg,Tg);
Tu = sdpvar(nu,nu,'diag');
Tu_mod = sdpvar(nu,nu,'diag');
Tu2 = blkdiag(Tu,Tu_mod);
Tz = sdpvar(n,n,'diag');

Y = sdpvar(n,n);
Y0 = sdpvar(n,nu);
Yfu = sdpvar(ny,nu);
Yf = sdpvar(ny,n);
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
eps =1*1e-12;
F = [F,gamma >= eps];

Mat2 = performance+bounds+robustness;

%%
n_mat = max(size(Mat2));
eps =1*1e-8;
F = [F, Mat2 <= -eps*eye(n_mat)];

%% Impose the constraints on the nn weights and biases;
F = [F,reshape(Y0-Tz*Wu,nu*n,1)<= reshape(Tz*tol_eps*ones(n,nu),nu*n,1)];
F = [F,reshape(Y0-Tz*Wu,nu*n,1)>= -reshape(Tz*tol_eps*ones(n,nu),nu*n,1)];

F = [F,reshape(Yf-Tg*Wf,1,n*ny)<= reshape(Tg*ones(ny,n)*tol_eps,1,ny*n)]; F = [F,reshape(Yf-Tg*Wf,1,n*ny)>= -reshape(Tg*ones(ny,n)*tol_eps,1,ny*n)];
F = [F,reshape(Yfu-Tg*Wfu,1,nx*ny)<=reshape(Tg*ones(ny,nu)*tol_eps,1,ny*nu)]; F = [F,reshape(Yfu-Tg*Wfu,1,nx*ny)>= -reshape(Tg*ones(ny,nu)*tol_eps,1,ny*nu)];

F = [F,reshape(Y-Tz*W,n^2,1)<=reshape(Tz*ones(n,n)*tol_eps,n^2,1)];
F = [F,reshape(Y-Tz*W,n^2,1)>=-reshape(Tz*ones(n,n)*tol_eps,n^2,1)];

%%
obj= gamma;

opt_details.set_solver = 'mosek';
% opt_details.set_solver = 'sedumi';

sol_2 = solvesdp(F,obj,sdpsettings('solver',opt_details.set_solver));

gamma_val = value(gamma);

problem_sol = sol_2.problem;

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
























