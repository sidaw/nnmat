function Wlr = getLowRankOptimization(W, rank)
   [m,n] = size(W);
   
   
   [U, S, V] = svd(W, 'econ');
   S = S(1:rank, 1:rank);
   U = U(:, 1:rank);
   V = V(:, 1:rank);
   initparams = [U(:); V(:)];
   options.MaxIter = 100;
   options.progTol = 1e-5;
   params = minFunc(@lossfunc, initparams, options);
   L = reshape(params(1:m*rank), m, rank);
   R = reshape(params(m*rank+1:end), rank, n);
   Wlr = L*R;
   
   function [loss, grad] = lossfunc(optparams)
       L = reshape(optparams(1:m*rank), m, rank);
       R = reshape(optparams(m*rank+1:end), rank, n);
       diff = (L*R-W);
       gradL = diff*R';
       gradR = L'*diff;
       grad = [gradL(:); gradR(:)];
       loss = 0.5*sum(sum(diff.*diff));
       
   end
end
