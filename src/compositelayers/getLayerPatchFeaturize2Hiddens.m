function [LayerPatchFeaturize, info] = getLayerPatchFeaturize2Hiddens(opts)
sizeimg = parseOption(opts, 'sizeimg', [28, 28]);
sizepatch = parseOption(opts, 'sizepatch', [4, 4]);
sizestride = parseOption(opts, 'sizestride', [1]);
numhid = parseOption(opts, 'numhid', 300);
numhid = parseOption(opts, 'numhid2', 300);
numpose = parseOption(opts, 'numpose', 6);

patchgen = LayerImage2Patch(sizeimg, sizepatch, sizestride);
patchlayer = LayerPatches(patchgen.dimpatches, numhid, patchgen.numpatches);
patchact = LayerActivation(numhid, 'relu');

patchlayer2 = LayerPatches(numhid, numphid2, patchgen.numpatches);
patchact2 = LayerActivation(numhid2, 'relu');

patchlayer3 = LayerPatches(numhid2, numpose, patchgen.numpatches);
patchact3 = LayerActivation(numpose, 'relu');

patch2flat = LayerFlattenPatches(numpose, patchgen.numpatches);

Lfeatures = {};
Lfeatures{end+1} = patchgen;
Lfeatures{end+1} = patchlayer;
Lfeatures{end+1} = patchact;

Lfeatures{end+1} = patchlayer2;
Lfeatures{end+1} = patchact2;

Lfeatures{end+1} = patchlayer3;
Lfeatures{end+1} = patchact3;

Lfeatures{end+1} = patch2flat;

LayerPatchFeaturize = LayersSerial(Lfeatures{:});
info.numout = patch2flat.numout;

end