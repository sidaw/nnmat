function [LayerPatchFeaturize, info] = getLayerAlexCIFAR26error(opts)
sizeimg = parseOption(opts, 'sizeimg', [32, 32]);
sizepatch = parseOption(opts, 'sizepatch', [5, 5]);
sizestride = parseOption(opts, 'sizestride', [1]);
numfilter = parseOption(opts, 'numfilter', 32);
numchan = parseOption(opts, 'numchan', 3);

patchgen = LayerImage2Patch(sizeimg, sizepatch, sizestride, numchan);
patchlayer = LayerPatches(patchgen.dimpatches, numfilter, patchgen.numpatches);
patchact = LayerActivation(numfilter, 'relu');
sizeconv = sqrt(patchgen.numpatches)*[1,1];
poolinglayer = LayerSpatialMaxPooling(sizeconv, [3,3], 2);
numpatch = poolinglayer.numpatches;
optflat.transpose = 1;
patch2flat = LayerFlattenPatches(numfilter, numpatch, optflat);

sizepool1 = sqrt(numpatch)*[1,1];
patchgen2 = LayerImage2Patch(sizepool1, sizepatch, sizestride, numfilter);
patchlayer2 = LayerPatches(patchgen2.dimpatches, numfilter, patchgen2.numpatches);
patchact2 = LayerActivation(numfilter, 'relu');
sizeconv2 = sqrt(patchgen2.numpatches)*[1,1];
poolinglayer2 = LayerSpatialMaxPooling(sizeconv2, [3,3], 2);
numpatch2 = poolinglayer2.numpatches;
optflat.transpose = 1;
patch2flat2 = LayerFlattenPatches(numfilter, numpatch2, optflat);


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