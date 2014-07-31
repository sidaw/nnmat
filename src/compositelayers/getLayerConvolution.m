function [LayerPatchFeaturize, info] = getLayerConvolution(opts)
sizeimg = parseOption(opts, 'sizeimg', [32, 32]);
sizepatch = parseOption(opts, 'sizepatch', [5, 5]);
sizestride = parseOption(opts, 'sizestride', [1]);
numfilter = parseOption(opts, 'numfilter', 32);
numchan = parseOption(opts, 'numchan', 3);
activation = parseOption(opts, 'activation', 'relu');

patchgen = LayerImage2Patch(sizeimg, sizepatch, sizestride, numchan);
patchlayer = LayerPatches(patchgen.dimpatch, numfilter, patchgen.numpatch);
patchact = LayerActivation(numfilter, activation);

transpose = 1;
patch2flat = LayerFlattenPatches(numfilter, patchgen.numpatch, transpose);

Lfeatures = {};
Lfeatures{end+1} = patchgen;
Lfeatures{end+1} = patchlayer;
Lfeatures{end+1} = patchact;
Lfeatures{end+1} = patch2flat;
LayerPatchFeaturize = LayersSerial(Lfeatures{:});
info.numout = patch2flat.numout;
info.numpatch = patchgen.numpatch;
info.numchan = numchan;
% TODO: deal with non-square images...
info.sizeimg = [1,1]*sqrt(patchgen.numpatch);
end