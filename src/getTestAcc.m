function testacc = getTestAcc(w, nn, Xtest, ytest)
    nn.setparams(w);
    numexample = size(Xtest,2);
    batchsize = 200;
    testpreds = zeros(1, numexample);
    
    for startind=1:batchsize:numexample
        endind = min(startind + batchsize - 1, numexample);
        
        testsub = Xtest(:, startind: endind);
        [~, subpreds] = max(nn.forward(testsub),[],1);
        testpreds(startind:endind) = subpreds;
    end
    [~, testlabels] = max(ytest,[],1);
    testacc = mean(testlabels == testpreds);
end