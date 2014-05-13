datapath = '~/data/mnisty/';
X = loadMNISTImages([datapath 'train-images-idx3-ubyte']);
yraw = loadMNISTLabels([datapath 'train-labels-idx1-ubyte']);
y = to1ofk(yraw)'; 
Xtest = loadMNISTImages([datapath 't10k-images-idx3-ubyte']);
ytestraw = loadMNISTLabels([datapath 't10k-labels-idx1-ubyte']);
ytest = to1ofk(ytestraw)';

save([datapath 'mnist.mat'], 'X', 'y', 'Xtest', 'ytest');