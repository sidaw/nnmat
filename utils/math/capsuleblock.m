function Y = capsuleblock(X)
colsq = sum(X.*X,2);
colnorm = sqrt(colsq);
Y = bsxfun(@times, X, colnorm./(1+colsq)); 
end