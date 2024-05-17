function [u_ramp,res_norm] = MPC_iterations(D,Wf,Wfu,u0,xk,iters,S,w)

n = max(size(u0));
phi = compute_phi(u0);
y = u0;
    c_MPC = 1*S*xk+ 1*w;
    zeta = -1*c_MPC;
residual = y-D*phi-zeta;

for g = 1:iters
    c_MPC = 1*S*xk+ 1*w;
    zeta = -1*c_MPC;
    y_store (:,g) = y;
    phi_store (:,g) = phi; res_store (:,g) = residual;
    ykp1 = D*phi + zeta;

    phi = compute_phi(ykp1);
    y  = ykp1;

    residual = y-D*phi-zeta;
    res_norm(g) = norm(residual);
    res_norm_store(g)= norm(y-D*phi-zeta);
end

y_ramp = ykp1;
% u_ramp = -1*inv(H)*(F_add*xk+G'*phi);
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