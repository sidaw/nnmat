E = [];
ps = 50:10:300;

for p=ps
    p
    numsamp = 50000;
    samps = randn(numsamp, p);
    rownorms = sqrt(sum(samps.*samps,2));
    nsamps = bsxfun(@rdivide,samps,rownorms);
    E(end+1) = mean(mean(abs(nsamps(:,1:40)), 2), 1);
end
Proc = E .* sqrt(ps);

empr = Proc-sqrt(2/pi);
expo = 1;
coeflin = mean(empr .* power(ps, expo));
plot(ps, empr, 'r', ...
    ps, coeflin./power(ps, expo), 'b');
