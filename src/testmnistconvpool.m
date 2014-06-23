addpath(genpath('../utils'))
addpath(genpath('.'))

datapath = '~/data/mnisty/';
load([datapath 'mnist.mat'])

if ~testToolboxes('Parallel Computing Toolbox')
    castfunc = @(x) double(x);
    disp('No parallel computing toolbox')
else
    castfunc = @(x) gpuArray(single(x));
    disp('Using the GPU');
end

dimdata =  784;
numpatchhid = 16;

numhid = 100;
numclass = 10;
numdata = 60000;
droprate = 0.5;

pfopts.sizeimg = [28, 28]; pfopts.sizepatch = [5,5]; pfopts.sizestride = 1;
pfopts.numhid = [32]; pfopts.numhid2 = [64]; 
[LayerPatchFeaturize info] = getLayerPatch2ConvPool(pfopts);

L = {};
L{end+1} = LayerPatchFeaturize;
% the fully connected layers
numhid = 600;
L{end+1} = LayerLinear(info.numout, numhid);
L{end+1} = LayerActivation(numhid, 'relu');
noisinglayer = LayerNoising(0.5);
L{end+1} = noisinglayer;
L{end+1} = LayerLinear(numhid, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});

X = castfunc(X(1:dimdata, 1:numdata));

% datadroprate = 0;
% X = X .* (rand(size(X)) > datadroprate);
% Xtest = Xtest .* (rand(size(Xtest)) > datadroprate);

y = castfunc(y(:, 1:numdata));
params = castfunc(nn.getparams());

optsloss.lambdaL2 = 1e-7;
minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);
batchlossfunc = @(params) BatchLossFunction_DivideData(params, X, y, nn, 'nll_logprob', optsloss);

options.DerivativeCheck = 0;
options.BatchSize = 100;
options.MaxIter = 500;
options.eta = 1e-3;
options.PermuteData = 0;

statfunc = @(w) getTestAcc(w, nn, Xtest, ytest);
paramsopt = minFuncSGDMmtm(minibatchlossfunc, params, X, y, options, statfunc);
%%
options.MaxIter = 20;
paramsopt = minFunc(batchlossfunc, paramsopt, options);

% [~, trainpreds] = max(nn.forward(X),[],1);
% [~, trainlabels] = max(y,[],1);
% trainacc = mean(trainlabels == trainpreds);

tic
noisinglayer.testing = 1;
testacc = getTestAcc(paramsopt, nn, Xtest, ytest);
trainacc = getTestAcc(paramsopt, nn, X, y);
testtime = toc;
fprintf('testacc=%f\t trainacc=%f\t time=%f', testacc, trainacc, testtime)
