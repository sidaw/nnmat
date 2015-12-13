addpath(genpath('../utils'))

datapath = '~/data/mnisty/';
load([datapath 'mnist.mat'])

dimdata =  784;
numhid = 1024;
numclass = 10;
numhid2 = 1024;

L = {};
L{end+1} = LayerNoising(0.3);
L{end+1} = LayerLinear(dimdata, numhid);
L{end+1} = LayerActivation(numhid, 'relu');
L{end+1} = LayerNoising(0.5);
L{end+1} = LayerLinear(numhid, numhid2);
L{end+1} = LayerActivation(numhid, 'relu');
L{end+1} = LayerNoising(0.5);
L{end+1} = LayerLinear(numhid2, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});

numdata = 60000;

X = X(1:dimdata, 1:numdata);

y = castfunc(y(:, 1:numdata));
params = castfunc(nn.getparams());


minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob');

options.DerivativeCheck = 0;
options.BatchSize = 100;
options.MaxIter = 1000;
options.eta = 1e4;
options.PermuteData = 1;
options.RowMajor = 0;


optsloss.lambdaL2 = 1e-7;
minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);

paramsopt = minFuncSGDMmtm(minibatchlossfunc, params, X, y, options);

[~, trainpreds] = max(nn.forward(X),[],1);
[~, trainlabels] = max(y,[],1);
trainacc = mean(trainlabels == trainpreds);

[~, testpreds] = max(nn.forward(Xtest),[],1);
[~, testlabels] = max(ytest,[],1);
testacc = mean(testlabels == testpreds);
disp([trainacc testacc])
