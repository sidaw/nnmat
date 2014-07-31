function [LayerPatchFeaturize, info] = getLayerPooling(opts)
sizeimg = parseOption(opts, 'sizeimg', [16, 16]);
sizepatch = parseOption(opts, 'sizepatch', [3, 3]);
sizestride = parseOption(opts, 'sizestride', [2]);
numchan = parseOption(opts, 'numchan', 32);
pooltype = parseOption(opts, 'pooltype', 'max'); % accepts max, avg, and stoic for now

patchgen = LayerImage2Patch(sizeimg, sizepatch, sizestride, numchan);
poollayer = LayerAggregate(patchgen.dimpatch/numchan, numchan, patchgen.numpatch, pooltype);

Lfeatures = {};
Lfeatures{end+1} = patchgen;
Lfeatures{end+1} = poollayer;
LayerPatchFeaturize = LayersSerial(Lfeatures{:});
info.numpatch = patchgen.numpatch;
info.numout = patchgen.numpatch * numchan;
info.numchan = numchan;
info.sizeimg = [1,1] * sqrt(info.numpatch);
end