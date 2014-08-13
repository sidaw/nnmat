sizebatch = 2;
optsloss.lambdaL2 = 0;

options.DerivativeCheck = 1;
options.BatchSize = 100;
options.MaxIter = 50;
options.eta = 1e-4;
options.PermuteData = 0;


%% check image2patch
numchan = 3;
imsize = [10, 10];

datagenlayer = LayerGenerateFakeData([prod(imsize)*numchan, sizebatch]);
% arguments are imsize, block, step, numchan, options
testedlayer = LayerImage2Patch(imsize, [3,3], 2, 3);
nn = LayersSerial(datagenlayer, testedlayer);

y = randn(testedlayer.dimpatch * testedlayer.numchan, testedlayer.numpatch, sizebatch);
batchlossfunc = @(theta) BatchLossFunction(theta, 0, y, nn, 'gradcheckloss', optsloss);
theta0 = nn.getparams();
paramsopt = minFunc(batchlossfunc, theta0, options);


%% check layerpatches
numin = 9; numout = 5; numpatch = 6;
datagenlayer = LayerGenerateFakeData([numin, numpatch, sizebatch]);
% arguments are numin, numout, numpatches, options
testedlayer = LayerPatches(numin, numout, numpatch);
nn = LayersSerial(datagenlayer, testedlayer);

y = randn(numout, numpatch, sizebatch);
batchlossfunc = @(theta) BatchLossFunction(theta, 0, y, nn, 'gradcheckloss', optsloss);
theta0 = nn.getparams();
paramsopt = minFunc(batchlossfunc, theta0, options);

%% check aggregate layer, type = max or avg
aggnames = {'max', 'avg'};
for i = 1:2
aggname = aggnames{i};
numchan = 5;
sizepatch = 10;
numpatch = 7;

datagenlayer = LayerGenerateFakeData([sizepatch*numchan, numpatch, sizebatch]);
% arguments are self = LayerAggregate(sizepatch, numchan, numpatch, aggname)
testedlayer = LayerAggregate(sizepatch, numchan, numpatch, aggname);
nn = LayersSerial(datagenlayer, testedlayer);

y = randn(numchan * numpatch, sizebatch);
batchlossfunc = @(theta) BatchLossFunction(theta, 0, y, nn, 'gradcheckloss', optsloss);
theta0 = nn.getparams();
paramsopt = minFunc(batchlossfunc, theta0, options);
end

%% check layer activation
actnames = {'sigmoid', 'relu', 'negrelu', 'square', 'none', 'softmax', 'logsoftmax'};
for i = 1:length(actnames)
actname = actnames{i}
numhid = 20;

datagenlayer = LayerGenerateFakeData([numhid, sizebatch]);
% LayerActivation(numin, actname, options)
testedlayer = LayerActivation(numhid, actname);
nn = LayersSerial(datagenlayer, testedlayer);

y = randn(numhid, sizebatch);
batchlossfunc = @(theta) BatchLossFunction(theta, 0, y, nn, 'gradcheckloss', optsloss);
theta0 = nn.getparams();
paramsopt = minFunc(batchlossfunc, theta0, options);
end

%% check layer activation
actnames = {'sigmoid', 'relu', 'negrelu', 'square', 'none', 'softmax', 'logsoftmax', 'failgrad '};

for i = 1:length(actnames)
actname = actnames{i}
numhid = 20;

datagenlayer = LayerGenerateFakeData([numhid, sizebatch]);
% LayerActivation(numin, actname, options)
testedlayer = LayerActivation(numhid, actname);
nn = LayersSerial(datagenlayer, testedlayer);

y = randn(numhid, sizebatch);
batchlossfunc = @(theta) BatchLossFunction(theta, 0, y, nn, 'gradcheckloss', optsloss);
theta0 = nn.getparams();
paramsopt = minFunc(batchlossfunc, theta0, options);
end


%% check padding

imsize = [3, 3];
padsize = [1,1];
numchan = 3;

datagenlayer = LayerGenerateFakeData([prod(imsize)*numchan, sizebatch]);
% LayerPadImage(imsize, padsize, numchan)
testedlayer = LayerPadImage(imsize, padsize, numchan);
nn = LayersSerial(datagenlayer, testedlayer);

y = randn(prod(testedlayer.totalsize)*numchan, sizebatch);
batchlossfunc = @(theta) BatchLossFunction(theta, 0, y, nn, 'gradcheckloss', optsloss);
theta0 = nn.getparams();
paramsopt = minFunc(batchlossfunc, theta0, options);