function lse = logsumexp(X)
    base = max(X,[],1);
    X = bsxfun(@minus, X, base );
    lse = base + log(sum(exp(X),1));
end