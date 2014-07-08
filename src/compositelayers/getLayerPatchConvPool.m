function [LayerPatchFeaturize, info] = getLayerPatchConvPool(opts)
sizeimg = parseOption(opts, 'sizeimg', [28, 28]);
sizepatch = parseOption(opts, 'sizepatch', [4, 4]);
sizestride = parseOption(opts, 'sizestride', [1]);
numhid = parseOption(opts, 'numhid', 32);

patchgen = LayerImage2Patch(sizeimg, sizepatch, sizestride);
patchlayer = LayerPatches(patchgen.dimpatches, numhid, patchgen.numpatches);
patchact = LayerActivation(numhid, 'sigmoid');
poolinglayer = LayerSpatialMaxPooling(sqrt(patchgen.numpatches)*[1,1], [3,3], 2);
patch2flat = LayerFlattenPatches(numhid, patchgen.numpatches);

Lfeatures = {};
Lfeatures{end+1} = patchgen;
Lfeatures{end+1} = patchlayer;
Lfeatures{end+1} = patchact;
%Lfeatures{end+1} = poolinglayer;
Lfeatures{end+1} = patch2flat;
LayerPatchFeaturize = LayersSerial(Lfeatures{:});
info.numout = patch2flat.numout;

end