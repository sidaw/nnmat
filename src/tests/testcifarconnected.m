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

numhid = 1000;
numhid2 = 1000;
L = {};
noisedata = LayerNoising(0.3);
L{end+1} = noisedata;

L{end+1} = LayerLinear(dimdata, numhid);
L{end+1} = LayerActivation(numhid, 'relu');
noise1 = LayerNoising(droprate);
L{end+1} = noise1;

L{end+1} = LayerLinearPositive(numhid, numhid2);
L{end+1} = LayerActivation(numhid2, 'relu');
noise2 = LayerNoising(droprate);
L{end+1} = noise2;

L{end+1} = LayerLinearPositive(numhid, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});

X = castfunc(X(1:dimdata, 1:numdata));

% datadroprate = 0;
% X = X .* (rand(size(X)) > datadroprate);
% Xtest = Xtest .* (rand(size(Xtest)) > datadroprate);

y = castfunc(y(:, 1:numdata));
params = castfunc(nn.getparams());

optsloss.lambdaL2 = 5e-3;
minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);
batchlossfunc = @(params) BatchLossFunction_DivideData(params, X, y, nn, 'nll_logprob', optsloss);

options.DerivativeCheck = 0;
options.BatchSize = 100;
options.MaxIter = 50;
options.eta = 5e-4;
options.PermuteData = 0;

statfunc = @(w) 0; %getTestAcc(w, nn, Xtest, ytest);
paramsopt = minFuncSGDMmtm(minibatchlossfunc, params, X, y, options, statfunc);
%%
noise1.testing = 1; noise2.testing = 1; noisedata.testing = 1;
noisinglayer.testing = 0;
options.MaxIter = 10;
%paramsopt = minFunc(batchlossfunc, params, options);

% [~, trainpreds] = max(nn.forward(X),[],1);
% [~, trainlabels] = max(y,[],1);
% trainacc = mean(trainlabels == trainpreds);

tic
noisedata.testing = 1;
noise1.testing = 1; noise2.testing = 1;
testacc = getTestAcc(paramsopt, nn, Xtest, ytest);
trainacc = getTestAcc(paramsopt, nn, X, y);
testtime = toc;
fprintf('testacc=%f\t trainacc=%f\t time=%f', testacc, trainacc, testtime)
