% functions here should operate on k by n matrices
% where k is data dim, and n number of data
% x is the k-by-n input, and y the corresponding output
% the gradfunc states how to backpropagate through each particular
% activation

function [lossfunc, gradfunc] = getLoss(name)
switch name
    % simple activations, where y_i = f(x_i) independent of x_j
    case 'nll_logprob'
        lossfunc = @(p, y) -sum(y.*p, 1);
        gradfunc = @(p, y) -y;
        
    case 'nll'
        lossfunc = @(p, y) -sum(y.*log(p), 1);
        gradfunc = @(p, y) -y./p;
        
    otherwise
        lossfunc = @(p, y) 0.5*sum( (p-y).*(p-y), 1);
        gradfunc = @(p, y) (p-y);
end

end