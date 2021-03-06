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
numhid = 1024;
numclass = 10;
numdata = 60000;
numhid2 = 1024;
droprate = 0.0;
datadroprate = 0.0;

L = {};
noise1 = LayerNoising(datadroprate);
% L{end+1} = noise1;
L{end+1} = LayerLinear(dimdata, numhid);
L{end+1} = LayerActivation(numhid, 'relu');
noise2 = LayerNoising(droprate);
% L{end+1} = noise2;
L{end+1} = LayerLinear(numhid, numhid2);
L{end+1} = LayerActivation(numhid, 'relu');
noise3 = LayerNoising(droprate);
% L{end+1} = noise3;
L{end+1} = LayerLinear(numhid2, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});

X = castfunc(X(1:dimdata, 1:numdata));
y = castfunc(y(:, 1:numdata));
params = castfunc(nn.getparams());

optsloss.lambdaL2 = 1e-3;
minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);
batchlossfunc = @(params) BatchLossFunction(params, X, y, nn, 'nll_logprob');

options.DerivativeCheck = 0;
options.BatchSize = 50;
options.MaxIter = 100;
options.eta = 1e-2;
options.PermuteData = 1;
options.RowMajor = 0;
statfunc = @(w) getTestAcc(w, nn, Xtest, ytest);
paramsopt = minFuncAdagrad(minibatchlossfunc, params, X, y, options, statfunc);

%paramsopt = minFunc(batchlossfunc, params, options);

noise1.testing = 1; noise2.testing = 1; noise3.testing = 1;
%[~, trainpreds] = max(nn.forward(X),[],1);
%[~, trainlabels] = max(y,[],1);
% trainacc = mean(trainlabels == trainpreds);

[~, testpreds] = max(nn.forward(Xtest),[],1);
[~, testlabels] = max(ytest,[],1);
testacc = mean(testlabels == testpreds);
disp([testacc])
