addpath(genpath('../utils'))

datapath = '~/data/mnisty/';
load([datapath 'mnist.mat'])

dimdata =  784;
numhid = 1024;
numclass = 10;
numdata = 60000;
numhid2 = 1024;
droprate = 0.5;
datadroprate = 0.3;

L = {};
noise1 = LayerNoising(datadroprate);
L{end+1} = noise1;
L{end+1} = LayerLinear(dimdata, numhid);
L{end+1} = LayerActivation(numhid, 'relu');
noise2 = LayerNoising(droprate);
L{end+1} = noise2;
L{end+1} = LayerLinear(numhid, numhid2);
L{end+1} = LayerActivation(numhid, 'relu');
noise3 = LayerNoising(droprate);
L{end+1} = noise3;
L{end+1} = LayerLinear(numhid2, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});

X = gpuArray(X(1:dimdata, 1:numdata));
y = gpuArray(y(:, 1:numdata));
params = gpuArray(nn.getparams());

minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob');
batchlossfunc = @(params) BatchLossFunction(params, X, y, nn, 'nll_logprob');

options.DerivativeCheck = 0;
options.BatchSize = 100;
options.MaxIter = 2000;
options.eta = 1e-2;
options.PermuteData = 1;
options.RowMajor = 0;
paramsopt = minFuncAdagrad(minibatchlossfunc, params, X, y, options);

%paramsopt = minFunc(batchlossfunc, params, options);

noise1.testing = 1; noise2.testing = 1; noise3.testing = 1;
[~, trainpreds] = max(nn.forward(X),[],1);
[~, trainlabels] = max(y,[],1);
trainacc = mean(trainlabels == trainpreds);

[~, testpreds] = max(nn.forward(Xtest),[],1);
[~, testlabels] = max(ytest,[],1);
testacc = mean(testlabels == testpreds);
disp([trainacc testacc])
