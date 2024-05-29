function y = ReLU(x)
% implements the ReLU nonlinearity
n = max(size(x));

y = x;

for j = 1:n
    if x(j)<=0
        y(j) = 0;
    end
end

end