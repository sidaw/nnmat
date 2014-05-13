function [w, finalObj] = minFuncSGD(funObj,  W, X, y, options)
% accepts only column major data as it should be in matlab
% that is, each data point occupies a single column
eta = options.eta;
w = W;
numdata = size(X,2);

fprintf('Batchsize:%d\tMaxIter:%d\tNumdata:%d\n', ...
    options.BatchSize, options.MaxIter, numdata)
iter = 1;
for t = 1:options.MaxIter
    batchobj = 0;
    rng(1)
    if options.PermuteData
        perm = randperm(length(y));
        X = X(:, perm);
        y = y(:, perm);
    end
    
    for b = 1:ceil(numdata/options.BatchSize)
        select = (b-1)* options.BatchSize+1:min(b* options.BatchSize, numdata);
        [finalObj, g] = funObj(w,X(:, select), y(:, select));
        
        batchobj = batchobj + finalObj;
        
        w = w - eta./sqrt(iter)*g;
        iter = iter + 1;
    end
    fprintf('%d\t%f\t%f\n', t, batchobj, norm(g))
end
end