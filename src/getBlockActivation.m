% functions here should operate on k by n matrices
% where k is data dim, and n number of data
% x is the k-by-n input, and y the corresponding output
% the gradfunc states how to backpropagate through each particular
% activation

function [actfunc, gradfunc] = getBlockActivation(name, options)
switch name
    % simple activations, where y_i = f(x_i) independent of x_j
    case 'capsulev2'
        actfunc = @capsuleblock;
        gradfunc = @capsuleblockgrad;
    
    % "normalizing" activations, where y_i = f(x_i, g(x_1, ..., x_k) )
    % quite commonly used, and no need to deal with the full jacobian
    case 'softmax'
        actfunc = @softmax;
        gradfunc = @(x,y,dfdo)  dfdo.*y - bsxfun(@times, y, sum(dfdo.*y, 1));
        
    case 'logsoftmax'
        actfunc = @logsoftmax;
        gradfunc = @(x,y,dfdo) (dfdo - bsxfun(@times, exp(y), sum(dfdo, 1)));
        % if the next layer is NLL, dfdo is just a 1-of-k encoding of the label
        
    otherwise
        actfunc = @(x) x;
        gradfunc = @(x,y,dfdo) dfdo;
end

end