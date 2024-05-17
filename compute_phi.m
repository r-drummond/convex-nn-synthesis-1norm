function phi = compute_phi(y)

ny = max(size(y));
phi = y;

for j = 1:ny
    if y(j)<= 0
        phi(j) = 0;        
    end
end