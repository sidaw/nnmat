E = [];
ps = 1:1:100;
for p=ps
    p
    numsamp = 100;
    samps = randn(numsamp, p);
    rownorms = sqrt(sum(samps.*samps,2));
    nsamps = bsxfun(@rdivide,samps,rownorms);
    E(end+1) = mean(mean(abs(nsamps), 1));
end
Proc = E .* sqrt(ps);

empr = Proc-sqrt(2/pi);
coeflin = mean(empr .* ps);
plot(ps, empr, 'r', ...
    ps, coeflin./ps, 'b');
