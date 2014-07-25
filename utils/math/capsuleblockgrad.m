function grad = capsuleblockgrad(X, Y, dfdo)
colsq = sum(X.*X,2);
colnorm = sqrt(colsq);

s = colnorm ./ (1+colsq);
dsdx = bsxfun(@rdivide, X, ((1+colsq).*colnorm)) - ... 
    bsxfun(@times, 2*X, colnorm ./ (1+colsq).^2);
consts = sum(X.*dfdo,2);
grad = bsxfun(@times, consts, dsdx) + bsxfun(@times, s, dfdo);
end