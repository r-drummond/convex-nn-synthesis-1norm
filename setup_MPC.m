function [D,S,w,N,nx,H,F,G,A,B]  = setup_MPC()

n = 3; nx = 2*n;
scaling = 1*1e0;

A = [4/3, -2/3;1 0]; B =[0;1];
N = 2;

Rtilde = 1; Ptilde = [7.1667,-4.2222;-4.2222 , 4.6852]; Qtilde = [1,-2/3;-2/3 , 3/22];

R = Rtilde; P = Ptilde;
for j = 1:n-1
    R = blkdiag(R,Rtilde); P = blkdiag(Qtilde,P);
end

My = zeros(N*n,n);
Gtilde = [0.1;-0.1]; G = Gtilde;
for i = 1:n
    for j = 1:i
        My((i-1)*(N)+1:i*N,j) = (A^(i-j))*B;
    end
    if i<n
        G = blkdiag(G,Gtilde);
    end
end

H = R + My'*P*My;
F_add_quadprog = 1*[A'*Qtilde*B,zeros(N,n-1)]';
F_add = F_add_quadprog*1;
S = G*(inv(H))*F_add;

F = F_add;
D = eye(2*n)-G*inv(H)*G';
w = ones(2*n,1)*scaling;


end