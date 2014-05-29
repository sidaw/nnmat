function [w, finalObj] = minFuncExpAdaptSignGrad(funObj,  W, X, y, options, funcStat)
% accepts only column major data as it should be in matlab
% that is, each data point occupies a single column
if nargin < 6
    funcStat = @(x) '';
end

eta = options.eta;
w = W;
numdata = size(X,2);
G = 1e-5*ones(size(W));
fprintf('Batchsize:%d\tMaxIter:%d\tNumdata:%d\n', ...
    options.BatchSize, options.MaxIter, numdata)

rate = ones(size(W));
prevsign = zeros(size(W));
adaptrate = 1.1;
decayrate = 2;
for t = 1:options.MaxIter
    batchobj = 0;
    %rng(1)
    if options.PermuteData
        perm = randperm(length(y));
        X = X(:, perm);
        y = y(:, perm);
    end
    
    for b = 1:ceil(numdata/options.BatchSize)
        select = (b-1)* options.BatchSize+1:min(b* options.BatchSize, numdata);
        [finalObj, g] = funObj(w,X(:, select), y(:, select));
        
        batchobj = batchobj + finalObj;
        
        signg = sign(g);
        w = w - eta * signg.* rate;
        signchanged = (prevsign&signg) & (prevsign ~= signg);
        signunchanged = (prevsign .* signg) == -1;
        rate(signchanged) = rate(signchanged) / decayrate;
        rate(signunchanged) = rate(signunchanged) * adaptrate;
        
        prevsign = signg;
        
    end
    statstring = funcStat(w);
    fprintf('%d\t%f\t%f\t statstring: %s\n', t, batchobj, norm(g), statstring)
end
end
