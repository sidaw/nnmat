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

numhid = 100;
numclass = 10;
numdata = 100;
droprate = 0.5;

pfopts.sizeimg = [28, 28]; pfopts.sizepatch = [5,5]; pfopts.sizestride = 1;
pfopts.numhid = [32]; pfopts.numhid2 = [64]; 
[LayerPatchFeaturize info] = getLayerPatchFeaturize(pfopts);

L = {};
L{end+1} = LayerPatchFeaturize;
% the fully connected layers
numhid = 300;
% L{end+1} = LayerLinear(info.numout, numhid);
% L{end+1} = LayerActivation(numhid, 'sigmoid');
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
batchlossfunc = @(params) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);

options.DerivativeCheck = 1;
options.BatchSize = 100;
options.MaxIter = 10;
options.eta = 5e-2;
options.PermuteData = 0;

statfunc = @(w) getTestAcc(w, nn, Xtest, ytest);
%paramsopt = minFuncAdagrad(minibatchlossfunc, params, X, y, options, statfunc);

paramsopt = minFunc(batchlossfunc, params, options);
% noise.testing = 1;
% [~, trainpreds] = max(nn.forward(X),[],1);
% [~, trainlabels] = max(y,[],1);
% trainacc = mean(trainlabels == trainpreds);

tic
testacc = getTestAcc(paramsopt, nn, Xtest, ytest);
trainacc = getTestAcc(paramsopt, nn, X, y);
testtime = toc;
fprintf('testacc=%f\t trainacc=%f\t time=%f', testacc, trainacc, testtime)
