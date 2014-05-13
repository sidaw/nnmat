function Wlr = getLowRank(W, rank)
   [U, S, V] = svd(W, 'econ');
   S = S(1:rank, 1:rank);
   U = U(:, 1:rank);
   V = V(:, 1:rank);
   Wlr = U*S*V';
end