function [w, finalObj] = minFuncSGD(funObj,  W, X, y, options)
eta = options.eta;
w = W;
numdata = size(X,1);
smooth = 20;
fprintf('Batchsize:%d\tMaxIter:%d\tNumdata:%d\n', ...
    options.BatchSize, options.MaxIter, numdata)
for t = 1:options.MaxIter
    batchobj = 0;
    
    if options.PermuteData
        perm = randperm(length(y));
        if options.RowMajor
            X = X(perm, :);
            y = y(perm, :);
        else
            X = X(:, perm);
            y = y(:, perm);
        end
    end
    
    for b = 1:ceil(size(X,1)/options.BatchSize)
        select = (b-1)* options.BatchSize+1:min(b* options.BatchSize, numdata);
        if options.RowMajor
            [finalObj, g] = funObj(w,X(select, :), y(select,:));
        else
            [finalObj, g] = funObj(w,X(:, select), y(:, select));
        end
        g = g / length(select);
        batchobj = batchobj + finalObj;
        w = w - eta*g/power(smooth/(t -1 + smooth),0.33);
    end
    fprintf('%d\t%f\t%f\n', t, batchobj, norm(g))
end
end
