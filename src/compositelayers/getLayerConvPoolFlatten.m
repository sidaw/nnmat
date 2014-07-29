function [LayerPatchFeaturize, info] = getLayerConvPoolFlatten(opts)
sizeimg = parseOption(opts, 'sizeimg', [32, 32]);
sizepatch = parseOption(opts, 'sizepatch', [5, 5]);
sizestride = parseOption(opts, 'sizestride', [1]);
numfilter = parseOption(opts, 'numfilter', 32);
numchan = parseOption(opts, 'numchan', 3);

sizepatchpool = parseOption(opts, 'sizepatchpool', [3, 3]);
sizestridepool = parseOption(opts, 'sizestridepool', [2]);

patchgen = LayerImage2Patch(sizeimg, sizepatch, sizestride, numchan);
patchlayer = LayerPatches(patchgen.dimpatches, numfilter, patchgen.numpatches);
patchact = LayerActivation(numfilter, 'relu');
sizeconv = sqrt(patchgen.numpatches)*[1,1];
poolinglayer = LayerSpatialMaxPooling(sizeconv, sizepatchpool, sizestridepool);
patch2flat = LayerFlattenPatches(numfilter, poolinglayer.numpatches, optflat);


Lfeatures = {};
Lfeatures{end+1} = patchgen;
Lfeatures{end+1} = patchlayer;
Lfeatures{end+1} = patchact;
Lfeatures{end+1} = poolinglayer;
Lfeatures{end+1} = patch2flat;
LayerPatchFeaturize = LayersSerial(Lfeatures{:});
info.numout = patch2flat.numout;

end