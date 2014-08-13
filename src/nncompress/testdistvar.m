E = [];
ps = 300:20:500;

for p=ps
    p
    numsamp = 10000;
    samps = randn(numsamp, p);
    rownorms = sqrt(sum(samps.*samps,2));
    nsamps = bsxfun(@rdivide,samps,rownorms);
    subindices = abs(nsamps(:,1));
    E(end+1) = var(subindices(:));
end

empr = E;
expo = 0.5;
coeflin = mean(empr .* power(ps, expo));
plot(ps, empr, 'r', ...
    ps, coeflin./power(ps, expo), 'b');
