function [ loss, grad ] = BatchLossFunction(params, X, y, nn, lossname, options)
    [lossfunc, lossgrad] = getLoss(lossname);
    nn.setparams(params);
    output = nn.forward(X);
    
    losses = lossfunc(output, y);
    
    
    lambda = parseOption(options, 'lambdaL2', 1e-6);
    loss = sum(losses(:)) + 0.5*lambda*sum(params.*params);
    
    dLdout = lossgrad(output, y);
    dldin = nn.backward(dLdout);
    
    grad = nn.getgrad() + lambda * params;
end

