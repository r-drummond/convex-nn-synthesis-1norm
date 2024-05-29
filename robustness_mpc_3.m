clc; close all; clear;
%% 29/05/24
% Code for "Convex neural network synthesis for robustness in the 1-norm"
% presented at the Learning for Dynamics and Control Conference, 2024.
% Authors: Ross Drummond, Chris Guiver, Matthew Turner.

% The code generates a neural network (NN) that approximates a model predictive control policy (MPC).
% The sedmi-definite programme optimises over the weights of the generated neural network to improve its robustness.
% A trade-off between the NN's accuracy relative to the MPC and its
% robustness (measured as per the definition in the paper) can be shown by
% tuning the parameter tol_eps.

% Requires the YALMIP toolbox and the MOSEK solver. 


%% Setup the MPC problem.
[D,S,w,N,nx,H,F,G,A,B]  = setup_MPC();
n = max(size(D));
nx = size(S,2);

n_mpc = size(G,2); %weights and biases of the MPC problem defined. See "Mapping back and forth between model predictive control and neural networks" by Drummond, Baldivieso-Monasterios and Valmorbida.
W = D;
Wu = S; b = w;
Wfu = -H\F;  Wf = -H\G';
F_store = F;
ny = size(Wfu,1);

Wmod = 1;

%%
bound_u = 1e2; % Define the input constraints
tol_eps = 1e-1; % Set the tolerance for the weight optimsiation. Epsilon from the paper. 
tol_eps = 1e-5;

[gamma_val,problem_sol,Y_val , Y0_val,Tz_val, Tg_val, W_val,Wu_val, Wf_val ,Wfu_val ] = compute_weights(D,S,w,N,nx,H,F,G,A,B,bound_u,tol_eps); % SDP to optimise over the weights.

%%
[gamma_val_orig,problem_sol_orig] = compute_gamma(W,Wu,Wf,Wfu,bound_u,tol_eps); % Compute the robutness bound for the MPC
[gamma_val_again,problem_sol_again] = compute_gamma(W_val,Wu_val,Wf_val,Wfu_val,bound_u,tol_eps); % Compute the robutness bound for the NN

gammas_all = [gamma_val_orig,gamma_val_again,gamma_val] % Compute the robustness bounds for both the MPC and the NN

%% Generate samples to compare the MPC and NN
n_samps = 1e2; % #of sampling points.
theta = 2*rand(n_samps,1); %initialise some vectors
iters = 5e3; u0 = ones(n,1);
u_ramp_store = zeros(n_mpc ,n_samps);
res_error = zeros(iters,n_samps); res_on_off = zeros(n_samps,1);
tol_res = 1e0;
for j = 1:n_samps
    x0(:,j) = bound_u*[cos(theta(j)*pi);sin(theta(j)*pi)]; % Sample some points for x[k]
    norm_u(j) = norm(x0(:,j));

    [u_ramp_store(:,j),res_norm] = MPC_iterations(D,Wf,Wfu,u0,x0(:,j),iters,Wu,b); % Run the MPC with the original weights and biases.
    u_action_store(j) =  u_ramp_store(1,j);

    [u_me_store(:,j),res_norm_me] = MPC_iterations(W_val,Wf_val,Wfu_val,u0,x0(:,j),iters,Wu_val,b); % Run the MPC with the optimised weights and biases.
    u_action_me_store(j) =  u_me_store(1,j);

    error_store(j) =  norm(u_action_me_store(j)- u_action_store(j),1); %compute the error

    res_store_mpc(j) = res_norm(end); res_store_me(j) = res_norm_me(end); %store the residuals of the implicit NN unravelling
    if res_store_me(j)<=tol_res
        res_on_off(j) = 1;
    end
end

%% Sample to see if the robustness bounds hold
u_mpc_max = max(u_action_me_store(:)); u_mpc_min = min(u_action_me_store(:));

for j = 1:n_samps
    if u_action_me_store(j) ==u_mpc_max
        x_max = x0(:,j);
    end
    if u_action_me_store(j) ==u_mpc_min
        x_min = x0(:,j);
    end

    value_max(j) = u_mpc_max-u_action_me_store(j); value_min(j) = u_action_me_store(j)-u_mpc_min;
    value(j) = max(value_max(j),value_min(j));
end


%% Run the MPC simulations
n_steps = 2e1; % #of steps
xk = bound_u/2*[1;-1]; %initial conditions
xk_me = xk;
uk = ones(n,1); uk_me = uk; % initialise vectors

for j = 1:n_steps
    [u_ramp_sim(:,j),res_norm] = MPC_iterations(D,Wf,Wfu,uk,xk,iters,Wu,b); % MPC with the original weights and biases
    u_sim(j) =  u_ramp_sim(1,j);
    res_end(j) = res_norm(end);

    [u_me_sim_set(:,j),res_norm_me] = MPC_iterations(W_val,Wf_val,Wfu_val,uk_me,xk_me,iters,Wu_val,b);  % MPC with the optimised weights and biases
    u_me_sim(j) =  u_me_sim_set(1,j);  res_end_me(j) = res_norm_me(end);

    xk = A*xk+B*u_sim(j); % perfrom the MPC state updates.
    xk_me = A*xk_me+B*u_me_sim(j);

    xk_store(:,j) = xk; xk_me_store(:,j) = xk_me; % store the updates
end

error_norm_x1 = norm(xk_store(1,:)-xk_me_store(1,j)); % compute the errors.
error_norm_x2 = norm(xk_store(2,:)-xk_me_store(2,j));
error_norm_u = norm(u_me_sim-u_sim);

%% Plot the results
close all
f_size = 24; f_size_leg = 18; f_size_gca = 18; f_size_axis = 16;


fig1 = figure;
subplot(2,1,1);hold on;
plot(1:n_steps, xk_store(1,:),'-xk','color',[0.2 0.2 0.2],'linewidth',1,'markersize',12);
plot(1:n_steps, xk_me_store(1,:),'-+k','color',0.8*[0.8 0.8 0.8],'linewidth',1,'markersize',12);
grid on
xlabel('Time step $k$','interpreter','latex','fontsize',f_size)
ylabel('$w_1[k]$','interpreter','latex','fontsize',f_size); box;
subplot(2,1,2);
hold on;
plot(1:n_steps, xk_store(2,:),'-xk','color',[0.2 0.2 0.2],'linewidth',1,'markersize',12);
plot(1:n_steps, xk_me_store(2,:),'-+k','color',0.8*[0.8 0.8 0.8],'linewidth',1,'markersize',12);
grid on
xlabel('Time step $k$','interpreter','latex','fontsize',f_size)
ylabel('$w_2[k]$','interpreter','latex','fontsize',f_size)
leg = legend('MPC','NN');
set(leg,'interpreter','latex','fontsize',f_size,'location','best')
box

fig4 = figure;
hold on;
plot(1:n_steps, xk_store(1,:),'-k','color',[0.2 0.2 0.2],'linewidth',1,'markersize',12);
plot(1:n_steps, xk_me_store(1,:),'-+k','color',0.8*[0.8 0.8 0.8],'linewidth',1,'markersize',12);
plot(1:n_steps, xk_store(1,:),'xk','color',[0.2 0.2 0.2],'linewidth',2,'markersize',15);
plot(1:n_steps, xk_me_store(1,:),'+k','color',0.8*[0.8 0.8 0.8],'linewidth',2,'markersize',15);
grid on
ax = gca;
ax.FontSize = f_size_gca;
xlabel('Time step $k$','interpreter','latex','fontsize',f_size)
ylabel('$w_1[k]$','interpreter','latex','fontsize',f_size)
box;
axis([0 n_steps, -50 100])

fig5 = figure;
hold on;
plot(1:n_steps, xk_store(2,:),'-k','color',[0.2 0.2 0.2],'linewidth',1,'markersize',12);
plot(1:n_steps, xk_me_store(2,:),'-+k','color',0.8*[0.8 0.8 0.8],'linewidth',1,'markersize',12);
plot(1:n_steps, xk_store(2,:),'xk','color',[0.2 0.2 0.2],'linewidth',2,'markersize',15);
plot(1:n_steps, xk_me_store(2,:),'+k','color',0.8*[0.8 0.8 0.8],'linewidth',2,'markersize',15);
grid on
ax = gca;
ax.FontSize = f_size_gca;
xlabel('Time step $k$','interpreter','latex','fontsize',f_size)
ylabel('$w_2[k]$','interpreter','latex','fontsize',f_size)
% leg = legend('MPC','NN');
% set(leg,'interpreter','latex','fontsize',f_size,'location','best'); 
% set(gca,'fontsize',f_size_axis )
box;
axis([0 n_steps, -60 120])

fig2 = figure;
hold on;
plot(1:n_steps, u_sim,'-xk','color',[0.2 0.2 0.2],'linewidth',1,'markersize',12);
plot(1:n_steps, u_me_sim,'-+k','color',0.8*[0.8 0.8 0.8],'linewidth',1,'markersize',12);
plot(1:n_steps, u_sim,'xk','color',[0.2 0.2 0.2],'linewidth',2,'markersize',15);
plot(1:n_steps, u_me_sim,'+k','color',0.8*[0.8 0.8 0.8],'linewidth',2,'markersize',15);
grid on
ax = gca;
ax.FontSize = f_size_gca;
xlabel('Time step $k$','interpreter','latex','fontsize',f_size)
ylabel('$v[k]$','interpreter','latex','fontsize',f_size)
leg = legend('MPC','Robustified NN');
set(leg,'interpreter','latex','fontsize',f_size_leg,'location','northeast'); box;



fig3 = figure;
hold on
plot(1:n_samps, gamma_val*ones(n_samps,1),'color',[0.2 0.2 0.2],'linewidth',2,'markersize',12);
plot(1:n_samps,value.*res_on_off','.b','color',[0.7 0.7 0.7],'linewidth',2,'markersize',12)
grid on
xlabel('Sample number','interpreter','latex','fontsize',f_size)
ylabel('$\|\tilde{u}\|_\infty$','interpreter','latex','fontsize',f_size)
leg = legend('Bound $\gamma$','Sample');
set(leg,'interpreter','latex','fontsize',f_size,'location','best')
box



% print(fig1,'xk_sim1em3','-depsc'); print(fig2,'uk_sim_1em3','-depsc');
% print(fig1,'xk_sim1em4','-depsc'); print(fig2,'uk_sim_1em4','-depsc');  print(fig2,'uk_sim_1em4','-dpng');
% print(fig1,'xk_sim1em5','-depsc'); print(fig2,'uk_sim_1em5','-depsc');  print(fig2,'uk_sim_1em5','-dpng');
% print(fig3,'samples','-depsc');

% print(fig2,'uk_sim_1em5','-depsc'); print(fig4,'x1_sim_1em5','-depsc'); print(fig5,'x2_sim_1em5','-depsc');
% print(fig2,'uk_sim_1em3','-depsc'); print(fig4,'x1_sim_1em3','-depsc'); print(fig5,'x2_sim_1em3','-depsc');
% print(fig2,'uk_sim_1em1','-depsc'); print(fig4,'x1_sim_1em1','-depsc'); print(fig5,'x2_sim_1em1','-depsc');
















