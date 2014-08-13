addpath(genpath('../utils'))
addpath(genpath('.'))

datapath = '~/data/cifar10/cifar-10-batches-mat/';
load([datapath 'cifarall.mat'])

if ~testToolboxes('Parallel Computing Toolbox')
    castfunc = @(x) double(x);
    disp('No parallel computing toolbox')
else
    castfunc = @(x) gpuArray(single(x));
    disp('Using the GPU');
end

dimdata =  3072;
numclass = 10;
numdata = 50000;

X = castfunc(X(1:dimdata, 1:numdata));
y = castfunc(y(:, 1:numdata));

%Xtest = castfunc(Xtest(1:dimdata, 1:numdata));
%ytest = castfunc(ytest(:, 1:numdata));
 
[nn] = getLayerAlexCIFAR26errorpad();
params = castfunc(nn.getparams());

optsloss.lambdaL2 = 1e-7;

minibatchlossfunc = @(params, X, y) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);
batchlossfunc = @(params) BatchLossFunction_DivideData(params, X, y, nn, 'nll_logprob', optsloss);

options.DerivativeCheck = 0;
options.BatchSize = 10;
options.MaxIter = 1000;
options.eta = 3e-2;
options.mom = 0.5;
options.PermuteData = 0;

statfunc = @(w) getTestAcc(w, nn, Xtest, ytest);
paramsopt = minFuncAdagrad(minibatchlossfunc, params, X, y, options, statfunc);
%%
%noisinglayer.testing = 0;
% options.MaxIter = 30;
paramsopt = minFunc(batchlossfunc, paramsopt, options);
% 
% [~, trainpreds] = max(nn.forward(X),[],1);
% [~, trainlabels] = max(y,[],1);
% trainacc = mean(trainlabels == trainpreds);

% tic
% %noisinglayer.testing = 1;
% testacc = getTestAcc(paramsopt, nn, Xtest, ytest);
% trainacc = getTestAcc(paramsopt, nn, X, y);
% testtime = toc;
% fprintf('testacc=%f\t trainacc=%f\t time=%f', testacc, trainacc, testtime)
