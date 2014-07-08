addpath(genpath('../utils'))

datapath = '~/data/mnisty/affnist_mat.mat';
load(datapath)

dimdata =  1600;
numhid = 1000;
numclass = 10;
numdata = 50000;

L = {};
L{1} = LayerLinear(dimdata, numhid);
L{2} = LayerActivation(numhid, 'relu');
L{3} = LayerLinear(numhid, numclass);
L{4} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});

X = batchdata_mat(1:dimdata, 1:numdata);
y = batchtargets_mat(:, 1:numdata);

params = nn.getparams();
lossfunc = @(params) BatchLossFunction(params, X, y, nn, 'nll_logprob');

options.DerivativeCheck = 0;
paramsopt = minFunc(lossfunc, params, options);

[~, trainpreds] = max(nn.forward(X),[],1);
[~, trainlabels] = max(y,[],1);
trainacc = mean(trainlabels == trainpreds);

Xtest = testbatchdata_mat;
ytest = testbatchtargets_mat;
[~, testpreds] = max(nn.forward(Xtest),[],1);
[~, testlabels] = max(ytest,[],1);
testacc = mean(testlabels == testpreds);