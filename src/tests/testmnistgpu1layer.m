addpath(genpath('../utils'))

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
numhid = 1200;
numclass = 10;
numdata = 60000;
droprate = 0.5;

L = {};
L{end+1} = LayerLinear(dimdata, numhid);
L{end+1} = LayerActivation(numhid, 'relu');

noise = LayerNoising(droprate);
noise.fixedmask = 0;
L{end+1} = noise;

L{end+1} = LayerLinear(numhid, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});

X = castfunc(X(1:dimdata, 1:numdata));

% datadroprate = 0;
% X = X .* (rand(size(X)) > datadroprate);
% Xtest = Xtest .* (rand(size(Xtest)) > datadroprate);

y = castfunc(y(:, 1:numdata));
params = castfunc(nn.getparams());

minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob');
batchlossfunc = @(params) BatchLossFunction(params, X, y, nn, 'nll_logprob');

options.DerivativeCheck = 0;
options.BatchSize = 100;
options.MaxIter = 2000;
options.eta = 1e-2;
options.PermuteData = 0;
paramsopt = minFuncAdagrad(minibatchlossfunc, params, X, y, options);

%paramsopt = minFunc(batchlossfunc, params, options);
noise.testing = 1;
[~, trainpreds] = max(nn.forward(X),[],1);
[~, trainlabels] = max(y,[],1);
trainacc = mean(trainlabels == trainpreds);

[~, testpreds] = max(nn.forward(Xtest),[],1);
[~, testlabels] = max(ytest,[],1);
testacc = mean(testlabels == testpreds);
disp([trainacc testacc])
