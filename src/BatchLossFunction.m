function [ loss, grad ] = BatchLossFunction(params, X, y, nn, lossname)
    [lossfunc, lossgrad] = getLoss(lossname);
    nn.setparams(params);
    output = nn.forward(X);
    
    losses = lossfunc(output, y);
    loss = sum(losses);
    
    dLdout = lossgrad(output, y);
    dldin = nn.backward(dLdout);
    
    grad = nn.getgrad();
end

