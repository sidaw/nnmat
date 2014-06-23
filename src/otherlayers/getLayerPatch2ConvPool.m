function [LayerPatchFeaturize, info] = getLayerPatch2ConvPool(opts)
sizeimg = parseOption(opts, 'sizeimg', [28, 28]);
sizepatch = parseOption(opts, 'sizepatch', [5, 5]);
sizestride = parseOption(opts, 'sizestride', [1]);
numhid = parseOption(opts, 'numhid', 32);
numhid2 = parseOption(opts, 'numhid2', 64);

patchgen = LayerImage2Patch(sizeimg, sizepatch, sizestride);
patchlayer = LayerPatches(patchgen.dimpatches, numhid, patchgen.numpatches);
patchact = LayerActivation(numhid, 'relu');
poolinglayer = LayerSpatialMaxPooling(sqrt(patchgen.numpatches)*[1,1], [3,3], 2);
 
patchlayer2 = LayerPatches(numhid, numhid2, poolinglayer.numpatches);
patchact2 = LayerActivation(numhid2, 'relu');
poolinglayer2 = LayerSpatialMaxPooling(sqrt(poolinglayer.numpatches)*[1,1], [2,2], 2);

patch2flat = LayerFlattenPatches(numhid, poolinglayer.numpatches);

Lfeatures = {};
Lfeatures{end+1} = patchgen;
Lfeatures{end+1} = patchlayer;
Lfeatures{end+1} = patchact;
Lfeatures{end+1} = poolinglayer;

% Lfeatures{end+1} = patchlayer2;
% Lfeatures{end+1} = patchact2;
% Lfeatures{end+1} = poolinglayer2;

Lfeatures{end+1} = patch2flat;
LayerPatchFeaturize = LayersSerial(Lfeatures{:});
info.numout = patch2flat.numout;

end