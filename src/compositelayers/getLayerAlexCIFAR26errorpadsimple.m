function [alexnn26] = getLayerAlexCIFAR26errorpad()

padlayer1 = LayerPadImage([32, 32], [2,2], 3);
conv1opts.sizeimg = [36, 36];
conv1opts.sizepatch = [5, 5];
conv1opts.sizestride = 1;
conv1opts.numfilter = 32;
conv1opts.numchan = 3;
conv1opts.activation = 'none';
[conv1, conv1info] = getLayerConvolution(conv1opts);

pool1opts.sizeimg = conv1info.sizeimg;
pool1opts.sizepatch = [3, 3];
pool1opts.sizestride = 2;
pool1opts.numchan = conv1opts.numfilter;
pool1opts.pooltype = 'max';
[pool1, pool1info] = getLayerPooling(pool1opts);


L = {};

L{end+1} = padlayer1;
L{end+1} = conv1;
L{end+1} = pool1;


numclass = 10;
L{end+1} = LayerLinear(pool1info.numout, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');

alexnn26 = LayersSerial(L{:});
end
