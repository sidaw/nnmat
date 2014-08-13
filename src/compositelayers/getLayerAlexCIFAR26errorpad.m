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

padlayer2 = LayerPadImage(pool1info.sizeimg, [2,2], pool1info.numchan);

conv2opts.sizeimg = padlayer2.totalsize;
conv2opts.sizepatch = [5, 5];
conv2opts.sizestride = 1;
conv2opts.numfilter = 32;
conv2opts.numchan = pool1info.numchan;
conv2opts.activation = 'relu';
[conv2, conv2info] = getLayerConvolution(conv2opts);

pool2opts.sizeimg = conv2info.sizeimg;
pool2opts.sizepatch = [3, 3];
pool2opts.sizestride = 2;
pool2opts.numchan = conv2opts.numfilter;
pool2opts.pooltype = 'avg';
[pool2, pool2info] = getLayerPooling(pool2opts);

padlayer3 = LayerPadImage(pool2info.sizeimg, [2,2], pool2info.numchan);

conv3opts.sizeimg = padlayer3.totalsize;
conv3opts.sizepatch = [5, 5];
conv3opts.sizestride = 1;
conv3opts.numfilter = 64;
conv3opts.numchan = pool2info.numchan;
conv3opts.activation = 'relu';
[conv3, conv3info] = getLayerConvolution(conv3opts);
% 
pool3opts.sizeimg = conv3info.sizeimg;
pool3opts.sizepatch = [3, 3];
pool3opts.sizestride = 2;
pool3opts.numchan = conv3opts.numfilter;
pool3opts.pooltype = 'avg';
[pool3, pool3info] = getLayerPooling(pool3opts);

L = {};

L{end+1} = padlayer1;
L{end+1} = conv1;
L{end+1} = pool1;

% L{end+1} = padlayer2;
% L{end+1} = conv2;
% L{end+1} = pool2;
% 
% L{end+1} = padlayer3;
% L{end+1} = conv3;
% L{end+1} = pool3;

numhid = 64;
L{end+1} = LayerLinear(pool1info.numout, numhid);
L{end+1} = LayerActivation(numhid, 'relu');

numclass = 10;
L{end+1} = LayerLinear(numhid, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');

alexnn26 = LayersSerial(L{:});
end
