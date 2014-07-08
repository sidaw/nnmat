addpath(genpath('../../utils'))
addpath(genpath('..'))

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

numclass = 10;
numdata = 100;
droprate = 0.5;

pfopts.sizeimg = [28, 28]; pfopts.sizepatch = [8,8]; pfopts.sizestride = 8;
pfopts.numhid = [10]; pfopts.numpose = [2]; 
[LayerPatchFeaturize, info] = getLayerPatchFeaturize(pfopts);

L = {};
L{end+1} = LayerPatchFeaturize;
% the fully connected layers

L{end+1} = LayerLinear(info.numout, numclass);

% optionally adds the noising layers
%noisinglayer = LayerNoising(0.5);
%L{end+1} = noisinglayer;
L{end+1} = LayerActivation(numclass, 'logsoftmax');

nn = LayersSerial(L{:});

X = castfunc(X(1:dimdata, 1:numdata));
y = castfunc(y(:, 1:numdata));

params = castfunc(nn.getparams());

optsloss.lambdaL2 = 1e-5;
minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);
batchlossfunc = @(params) BatchLossFunction_DivideData(params, X, y, nn, 'nll_logprob', optsloss);

options.DerivativeCheck = 1;
options.BatchSize = 100;
options.MaxIter = 100;
options.eta = 1e-7;
options.PermuteData = 0;

statfunc = @(w) 0; %getTestAcc(w, nn, Xtest, ytest);
paramsopt = minFuncSGDMmtm(minibatchlossfunc, params, X, y, options, statfunc);
%paramsopt = params;
%%

noisinglayer.testing = 0;
options.MaxIter = 100;
paramsopt = minFunc(batchlossfunc, paramsopt, options);

tic
noisinglayer.testing = 1;
testacc = getTestAcc(paramsopt, nn, Xtest, ytest);
trainacc = getTestAcc(paramsopt, nn, X, y);
testtime = toc;
fprintf('testacc=%f\t trainacc=%f\t time=%f', testacc, trainacc, testtime)
