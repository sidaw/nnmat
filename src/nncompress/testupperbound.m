n = 100;
p = 100;

vs = randn(n, p);
vsn = bsxfun(@rdivide, vs, sqrt(sum(vs.*vs,2)));
A = vsn * vsn';



diff = A - A';

assert(norm(diff(:)) < 1e-5)

cvx_begin sdp
  variables S(n,n)
  maximize trace(A*S)
  diag(S) == 1;
  S >= 0;
cvx_end

Anorm = sum(A(:).^2);
Snorm = sum(S(:).^2);