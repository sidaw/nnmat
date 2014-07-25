function [ loss, grad ] = BatchLossFunction_DivideData(params, X, y, nn, lossname, options)
    [lossfunc, lossgrad] = getLoss(lossname);
    nn.setparams(params);
    
    numexample = size(X,2);
    batchsize = 100;
    
    loss = 0;
    grad = 0;
    for startind=1:batchsize:numexample
        endind = min(startind + batchsize - 1, numexample);
        
        Xc = X(:, startind:endind);
        yc = y(:, startind:endind);
        outputc = nn.forward(Xc);
        lossesc = lossfunc(outputc, yc);
        loss = loss + sum(lossesc);
        
        dLdoutc = lossgrad(outputc, yc);
        dldinc = nn.backward(dLdoutc);
        grad = grad + nn.getgrad();
    
    end
   
    lambda = parseOption(options, 'lambdaL2', 1e-3);
    loss = loss + 0.5*lambda*sum(params.*params);
    grad = grad + lambda * params;
end

