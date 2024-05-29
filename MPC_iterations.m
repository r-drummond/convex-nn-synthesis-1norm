function [u_ramp,res_norm] = MPC_iterations(D,Wf,Wfu,u0,xk,iters,S,w)

%% this code computes the MPC policy by unravelling the implict neural network.
% See "Mapping back and forth between model predictive control and neural
% networks" by Drummond, Baldivieso-Monasterios and Valmorbida for details.

phi = compute_phi(u0);

for g = 1:iters
    c_MPC = 1*S*xk+ 1*w;
    zeta = -1*c_MPC;
    ykp1 = D*phi + zeta;

    phi = compute_phi(ykp1);
    y  = ykp1;

    residual = y-D*phi-zeta;
    res_norm(g) = norm(residual);
end

u_ramp = Wfu*xk+Wf*phi;

end

function phi = compute_phi(u)
n = max(size(u));
phi = zeros(n,1);
for g = 1:n
    if u(g) >0
        phi(g) = u(g);
    else
        phi(g) = 0;
    end
end
end