addpath(genpath('../utils'))

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
numhid = 100;
numclass = 10;
numdata =60000;

L = {};
%L{end+1} = LayerNoising(0.3);
L{end+1} = LayerLinear(dimdata, numhid);
L{end+1} = LayerBlockActivation(numhid, 'capsulev2', 10);

% switch this out to use normal activation
% L{end+1} = LayerActivation(numhid, 'relu');

%L{end+1} = LayerNoising(0.5);
L{end+1} = LayerLinear(numhid, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});

X = castfunc(X(1:dimdata, 1:numdata));
y = castfunc(y(:, 1:numdata));
params = castfunc(nn.getparams());

options.DerivativeCheck = 0;
options.BatchSize = 200;
options.eta = 1e-2;
options.PermuteData = 1;
options.RowMajor = 0;
optsloss.lambdaL2 = 0;
minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);
batchlossfunc = @(params) BatchLossFunction_DivideData(params, X, y, nn, 'nll_logprob', optsloss);

statfunc = @(w) getTestAcc(w, nn, Xtest, ytest);
options.MaxIter = 200;
%paramsopt = minFuncAdagrad(minibatchlossfunc, params, X, y, options, statfunc);

paramsopt = minFunc(batchlossfunc, params, options);

% [~, trainpreds] = max(nn.forward(X),[],1);
% [~, trainlabels] = max(y,[],1);
% trainacc = mean(trainlabels == trainpreds);

tic
noisinglayer.testing = 1;
testacc = getTestAcc(paramsopt, nn, Xtest, ytest);
trainacc = getTestAcc(paramsopt, nn, X, y);
testtime = toc;
fprintf('testacc=%f\t trainacc=%f\t time=%f', testacc, trainacc, testtime)
