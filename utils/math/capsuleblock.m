function Y = capsuleblock(X)
colsq = sum(X.*X,1);
colnorm = sqrt(colsq);
Y = bsxfun(@times, X, colnorm./(1+colsq)); 
end
