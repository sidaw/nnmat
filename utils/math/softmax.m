function expXwdZ = softmax(X)
X = bsxfun(@minus, X, max(X,[],1) );
expX = exp(X);
Z = sum(expX,1);
expXwdZ = bsxfun(@rdivide, expX, Z);
end