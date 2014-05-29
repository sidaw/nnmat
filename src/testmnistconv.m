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
numpatchhid = 16;

numhid = 800;
numclass = 10;
numdata = 60000;
droprate = 0.5;

L = {};
patchgen = LayerImage2Patch([28, 28], [4 4], 1);
patchlayer = LayerPatches(patchgen.dimpatches, numpatchhid, patchgen.numpatches);
patchact = LayerActivation(numpatchhid, 'relu');
patch2flat = LayerFlattenPatches(numpatchhid, patchgen.numpatches);

L{end+1} = patchgen;
L{end+1} = patchlayer;
L{end+1} = patchact;
L{end+1} = patch2flat;

L{end+1} = LayerLinear(patch2flat.dimout, numhid);
L{end+1} = LayerActivation(numhid, 'relu');

% noise = LayerNoising(droprate);
% noise.fixedmask = 0;
% L{end+1} = noise;

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
options.MaxIter = 10;
options.eta = 1e-2;
options.PermuteData = 0;

statfunc = @(w) getTestAcc(w, nn, Xtest, ytest);
paramsopt = minFuncAdagrad(minibatchlossfunc, paramsopt, X, y, options, statfunc);

%paramsopt = minFunc(batchlossfunc, params, options);
% noise.testing = 1;
[~, trainpreds] = max(nn.forward(X),[],1);
[~, trainlabels] = max(y,[],1);
trainacc = mean(trainlabels == trainpreds);

[~, testpreds] = max(nn.forward(Xtest),[],1);
[~, testlabels] = max(ytest,[],1);
testacc = mean(testlabels == testpreds);

disp([trainacc testacc])
