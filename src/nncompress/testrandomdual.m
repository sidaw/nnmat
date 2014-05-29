n = 100;
Ar = randn(n, n);

A = Ar + Ar';
A = Ar * Ar';

cvx_begin sdp
  variables u(n)
  minimize sum(u)
  
  diag(u) - A >= 0;
cvx_end

cvx_begin sdp
  variables S(n,n)
  maximize trace(A*S)
  diag(S) <= 1;
  S >= 0
cvx_end