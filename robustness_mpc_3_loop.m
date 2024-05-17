clc; close all; clear;

[D,S,w,N,nx,H,F,G,A,B]  = setup_MPC();
n = max(size(D));
nx = size(S,2);

n_mpc = size(G,2);
W = D;
Wu = S; b = w;
Wfu = -H\F;  Wf = -H\G';
F_store = F;
ny = size(Wfu,1);

Wmod = 1;
bound_u = 5e-1;
bound_u = 1e2;
tol_eps = 1e-5;



%%
[gamma_val_orig,problem_sol_orig] = compute_gamma(W,Wu,Wf,Wfu,bound_u,tol_eps);

n_loop = 1e2;
min_eps = -5;
max_eps = 1; 
eps_range = logspace(min_eps,max_eps,n_loop); counter = 1;
for j = 1:n_loop
    tol_eps = eps_range(j);
    [gamma_val,problem_sol,Y_val , Y0_val,Tz_val, Tg_val, W_val,Wu_val, Wf_val ,Wfu_val ] = compute_weights(D,S,w,N,nx,H,F,G,A,B,bound_u,tol_eps);
[gamma_val_again,problem_sol_again] = compute_gamma(W_val,Wu_val,Wf_val,Wfu_val,bound_u,tol_eps);
if gamma_val_again>1e-8
    gamma_store(counter) = gamma_val_again;
    eps_plot(counter) = tol_eps;
    counter = counter+1;
end
end

% gammas_all = [gamma_val_orig,gamma_val_again,gamma_val]

%% my_color = [0.9,0.9,0.9];
close all
f_size = 20; f_size_leg = 18; f_size_gca = 13;



fig1 = figure;
loglog(eps_plot,gamma_val_orig*ones(counter-1,1),'--k','color',[0.2 0.2 0.2],'linewidth',2,'markersize',12); hold on;
loglog(eps_plot,gamma_store,'-k','color',0.8*[0.8 0.8 0.8],'linewidth',2,'markersize',12); 
grid on
ax = gca;
ax.FontSize = f_size_gca;
xlabel('Tolerance constraining the weights: $\varepsilon$','interpreter','latex','fontsize',f_size)
ylabel('Robustness bound: $\gamma$','interpreter','latex','fontsize',f_size)
leg = legend('MPC','Robustified NN');
set(leg,'interpreter','latex','fontsize',f_size_leg,'location','southwest');
% box;
axis([10^min_eps, 10^max_eps , 10^-4 10^4])
xticks([1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1])
% xticklabels({'x = 0','x = 5','x = 10'})
% z

print(fig1,'trade_off','-depsc'); %print(fig4,'x1_sim_1em5','-depsc'); print(fig5,'x2_sim_1em5','-depsc');
% print(fig2,'uk_sim_1em3','-depsc'); print(fig4,'x1_sim_1em3','-depsc'); print(fig5,'x2_sim_1em3','-depsc');
% print(fig2,'uk_sim_1em1','-depsc'); print(fig4,'x1_sim_1em1','-depsc'); print(fig5,'x2_sim_1em1','-depsc');
















