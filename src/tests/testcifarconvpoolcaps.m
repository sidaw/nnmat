addpath(genpath('../utils'))
addpath(genpath('.'))

datapath = '~/data/cifar10/cifar-10-batches-mat/';
load([datapath 'cifarall.mat'])

if ~testToolboxes('Parallel Computing Toolbox')
    castfunc = @(x) single(x);
    disp('No parallel computing toolbox')
else
    castfunc = @(x) gpuArray(single(x));
    disp('Using the GPU');
end

dimdata =  3072;

numclass = 10;
numdata = 50000;
droprate = 0.5;

pfopts.sizeimg = [32, 32]; pfopts.sizepatch = [5,5]; pfopts.sizestride = 1;
pfopts.numhid = [32]; pfopts.numhid2 = [64]; 
pfopts.numchan = 3;
[LayerPatchFeaturize info] = getLayerPatch2ConvPool(pfopts);

L = {};
L{end+1} = LayerPatchFeaturize;
% the fully connected layers
numhid = 600;
L{end+1} = LayerLinear(info.numout, numhid);
%L{end+1} = LayerActivation(numhid, 'relu');
L{end+1} = LayerBlockActivation(numhid, 'capsulev2', 10);
%noisinglayer = LayerNoising(0.5);
%L{end+1} = noisinglayer;
L{end+1} = LayerBlockActivation(numhid, 'capsulev2', 10);

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
options.MaxIter = 300;
options.eta = 5e-4;
options.PermuteData = 0;

statfunc = @(w) getTestAcc(w, nn, Xtest, ytest);
paramsopt = minFuncSGDMmtm(minibatchlossfunc, params, X, y, options, statfunc);
%%
%noisinglayer.testing = 0;
%options.MaxIter = 10;
%paramsopt = minFunc(batchlossfunc, paramsopt, options);

% [~, trainpreds] = max(nn.forward(X),[],1);
% [~, trainlabels] = max(y,[],1);
% trainacc = mean(trainlabels == trainpreds);

tic
%noisinglayer.testing = 1;
testacc = getTestAcc(paramsopt, nn, Xtest, ytest);
trainacc = getTestAcc(paramsopt, nn, X, y);
testtime = toc;
fprintf('testacc=%f\t trainacc=%f\t time=%f', testacc, trainacc, testtime)
