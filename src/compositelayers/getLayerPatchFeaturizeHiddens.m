function [LayerPatchFeaturize, info] = getLayerPatchFeaturizeHiddens(opts)
sizeimg = parseOption(opts, 'sizeimg', [28, 28]);
sizepatch = parseOption(opts, 'sizepatch', [4, 4]);
sizestride = parseOption(opts, 'sizestride', [1]);
numhid = parseOption(opts, 'numhid', 300);
numpose = parseOption(opts, 'numpose', 6);

patchgen = LayerImage2Patch(sizeimg, sizepatch, sizestride);
patchlayer = LayerPatches(patchgen.dimpatches, numhid, patchgen.numpatches);
patchact = LayerActivation(numhid, 'relu');

patchlayer2 = LayerPatches(numhid, numpose, patchgen.numpatches);
patchact2 = LayerActivation(numpose, 'relu');

patch2flat = LayerFlattenPatches(numpose, patchgen.numpatches);

Lfeatures = {};
Lfeatures{end+1} = patchgen;
Lfeatures{end+1} = patchlayer;
Lfeatures{end+1} = patchact;
Lfeatures{end+1} = patchlayer2;
Lfeatures{end+1} = patchact2;
Lfeatures{end+1} = patch2flat;

LayerPatchFeaturize = LayersSerial(Lfeatures{:});
info.numout = patch2flat.numout;

end