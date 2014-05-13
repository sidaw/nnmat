testaccs = [];
trainaccs = [];
ranks = floor((1:35).^1.5);
load('firstlayerweight150error.mat');

for rank = ranks
Wlr = getLowRank(Wtrained, rank);
imilr = reshape(Wlr(i, :), 28, []);
imi = reshape(Wtrained(i, :), 28, []);
subplot(2,1,1); imagesc(imilr); subplot(2,1,2); imagesc(imi)

nn.layers{1}.setparams(Wlr);
[~, trainpreds] = max(nn.forward(X),[],1);
[~, trainlabels] = max(y,[],1);
trainacc = mean(trainlabels == trainpreds);

[~, testpreds] = max(nn.forward(Xtest),[],1);
[~, testlabels] = max(ytest,[],1);
testacc = mean(testlabels == testpreds);
disp([trainacc testacc])

trainaccs(end+1) = trainacc;
testaccs(end+1)=testacc;

end


plot(ranks, trainaccs, 'b', ranks, testaccs, 'r');
