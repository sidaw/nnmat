datapath = '~/data/mnisty/';
X = loadMNISTImages([datapath 'train-images-idx3-ubyte']);
yraw = loadMNISTLabels([datapath 'train-labels-idx1-ubyte']);
y = to1ofk(yraw)'; 
Xtest = loadMNISTImages([datapath 't10k-images-idx3-ubyte']);
ytestraw = loadMNISTLabels([datapath 't10k-labels-idx1-ubyte']);
ytest = to1ofk(ytestraw)';

sizepatch = [8, 8];
sizestride = 4;

Xpatch = im2patch(X, [28,28], sizepatch, sizestride);
Xtestpatch = im2patch(Xtest, [28,28], sizepatch, sizestride);

save([datapath 'mnistpatch.mat'], 'Xpatch', 'y', 'Xtestpatch', 'ytest');