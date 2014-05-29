function testacc = getTestAcc(w, nn, Xtest, ytest)
    nn.setparams(w);
    [~, testpreds] = max(nn.forward(Xtest),[],1);
    [~, testlabels] = max(ytest,[],1);
    testacc = mean(testlabels == testpreds);
end