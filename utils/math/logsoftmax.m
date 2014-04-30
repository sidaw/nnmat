function ls = logsoftmax(X)
    ls = bsxfun(@minus, X, logsumexp(X) );
end