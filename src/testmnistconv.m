addpath(genpath('../utils'))
addpath(genpath('.'))

datapath = '~/data/mnisty/';
load([datapath 'mnist.mat'])

if ~testToolboxes('Parallel Computing Toolbox')
    castfunc = @(x) single(x);
    disp('No parallel computing toolbox')
else
    castfunc = @(x) gpuArray(single(x));
    disp('Using the GPU');
end

dimdata =  784;
numpatchhid = 16;

numhid = 800;
numclass = 10;
numdata = 60000;
droprate = 0.5;

pfopts.sizeimg = [28, 28]; pfopts.sizepatch = [8,8]; pfopts.sizestride = 2;
pfopts.numhid = [50]; pfopts.numhid2 = [30]; ptopts.numpose = 10;
[LayerPatchFeaturize info] = getLayerPatchFeaturize(pfopts);

L = {};

% the fully connected layers
L{end+1} = LayerPatchFeaturize;
% L{end+1} = LayerLinear(info.numout, numhid);
% L{end+1} = LayerActivation(numhid, 'relu');

% noise = LayerNoising(droprate);
% noise.fixedmask = 0;
% L{end+1} = noise;

L{end+1} = LayerLinear(info.numout, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});

X = castfunc(X(1:dimdata, 1:numdata));

% datadroprate = 0;
% X = X .* (rand(size(X)) > datadroprate);
% Xtest = Xtest .* (rand(size(Xtest)) > datadroprate);

y = castfunc(y(:, 1:numdata));
params = castfunc(nn.getparams());

optsloss.lambdaL2 = 1e-3;
minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);
batchlossfunc = @(params) BatchLossFunction(params, X, y, nn, 'nll_logprob');

options.DerivativeCheck = 0;
options.BatchSize = 100;
options.MaxIter = 30;
options.eta = 5e-2;
options.PermuteData = 0;

statfunc = @(w) getTestAcc(w, nn, Xtest, ytest);
paramsopt = minFuncAdagrad(minibatchlossfunc, params, X, y, options, statfunc);

%paramsopt = minFunc(batchlossfunc, params, options);
% noise.testing = 1;
% [~, trainpreds] = max(nn.forward(X),[],1);
% [~, trainlabels] = max(y,[],1);
% trainacc = mean(trainlabels == trainpreds);

tic
[~, testpreds] = max(nn.forward(Xtest),[],1);
[~, testlabels] = max(ytest,[],1);
testacc = mean(testlabels == testpreds);
testtime = toc;
fprintf('testacc=%f\t time=%f', testacc, testtime)
