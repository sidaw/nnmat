
if ~testToolboxes('Parallel Computing Toolbox')
    castfunc = @(x) single(x);
    disp('No parallel computing toolbox')
else
    castfunc = @(x) gpuArray(single(x));
    disp('Using the GPU');
end
options.DerivativeCheck = 1;
options.BatchSize = 100;
options.MaxIter = 10;
options.eta = 5e-2;
options.PermuteData = 0;

%% Test gradient for logistic regression
X = castfunc(randn(4,5));
y = castfunc( to1ofk([1 2 3 3 2])' );

L = {};
L{end+1} = LayerLinear(4, 3);
L{end+1} = LayerActivation(3, 'logsoftmax');
nn = LayersSerial(L{:});
params = castfunc(nn.getparams());
optsloss.lambdaL2 = 1e-3;
batchlossfunc = @(params) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);

paramsopt = minFunc(batchlossfunc, params, options);


%%
%% Test gradient for logistic regression
inputdim = 10;
numhid = 5;
X = castfunc(randn(inputdim,5));
y = castfunc( to1ofk([1 2 3 3 2])' );

L = {};
L{end+1} = LayerLinear(inputdim, numhid);
L{end+1} = LayerActivation(numhid, 'sigmoid');
L{end+1} = LayerLinear(numhid, 3);
L{end+1} = LayerActivation(3, 'logsoftmax');

nn = LayersSerial(L{:});
params = castfunc(nn.getparams());
optsloss.lambdaL2 = 1e-3;
batchlossfunc = @(params) BatchLossFunction(params, X, y, nn, 'nll_logprob', optsloss);

paramsopt = minFunc(batchlossfunc, params, options);