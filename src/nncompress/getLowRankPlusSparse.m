function [Wlr Wsparse] = getLowRankPlusSparse(W, coeff)
   [m,n] = size(W);
   eps = 1e-1;
   run /usr/local/google/home/sidaw/matlib/cvx/cvx_startup.m
   %cvx_solver sdpt3;
   cvx_begin sdp
     variables Wlr(m,n)
     variables Wsparse(m,n)
     minimize coeff*norm_nuc(Wlr) + norm(Wsparse(:), 1)
        norm(Wlr + Wsparse - W) <= eps;
   cvx_end
end
