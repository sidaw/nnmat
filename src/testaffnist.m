addpath(genpath('../utils'))

datapath = '~/data/mnisty/';
load([datapath 'affnist.mat'])

dimdata =  1600;
numhid = 1048;
numclass = 10;
numdata = 50000;
numhid2 = 1048;

L = {};
L{1} = LayerLinear(dimdata, numhid);
L{end+1} = LayerActivation(numhid, 'relu');
L{end+1} = LayerLinear(numhid, numhid2);
L{end+1} = LayerActivation(numhid, 'relu');
L{end+1} = LayerLinear(numhid2, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});

X = X(1:dimdata, 1:numdata);
y = y(:, 1:numdata);

params = nn.getparams();
minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob');

options.DerivativeCheck = 0;
options.BatchSize = 100;
options.MaxIter = 300;
options.eta = 1e-2;
options.PermuteData = 1;
options.RowMajor = 0;
paramsopt = minFuncAdagrad(minibatchlossfunc, params, X, y, options);

[~, trainpreds] = max(nn.forward(X),[],1);
[~, trainlabels] = max(y,[],1);
trainacc = mean(trainlabels == trainpreds);

[~, testpreds] = max(nn.forward(Xtest),[],1);
[~, testlabels] = max(ytest,[],1);
testacc = mean(testlabels == testpreds);
