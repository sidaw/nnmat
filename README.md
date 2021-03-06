# nnmat

Just an experimental deep learning library in object oriented matlab.
To run a 2 hidden layers network with dropout on mnist, you simply need the following:

```matlab
dimdata =  784;
numhid = 1024;
numclass = 10;
numhid2 = 1024;

L = {};
L{end+1} = LayerNoising(0.3);
L{end+1} = LayerLinear(dimdata, numhid);
L{end+1} = LayerActivation(numhid, 'relu');
L{end+1} = LayerNoising(0.5);
L{end+1} = LayerLinear(numhid, numhid2);
L{end+1} = LayerActivation(numhid, 'relu');
L{end+1} = LayerNoising(0.5);
L{end+1} = LayerLinear(numhid2, numclass);
L{end+1} = LayerActivation(numclass, 'logsoftmax');
nn = LayersSerial(L{:});
```

The data should be in a matrix of datadim by numhid. See src/tests/testconvextrain.m or src/tests/testmnist.m for the whole thing.
nnmat followed the torch/nn style of design, and can run on gpu if set up correctly.

## how to run
```matlab
cd scr
initnnmat
testconvextrain
```
or look at any of these testX.m

## sigmoid vs. relu

It appears that relu and sigmoid units can make a big difference. This is well-known but poorly understood.
Try:
```matlab
initnnmat;testconvextrain; plotObjectivePermute
```
Comment out the sigmoid layer and uncomment the relu line

```matlab
L{end+1} = LayerActivation(numhid, 'sigmoid');
%L{end+1} = LayerActivation(numhid, 'relu');
```

Run this again: 
```matlab
testconvextrain; plotObjectivePermute
```

The expected outputs are [the sigmoid figure](src/plots/sigmoid.pdf) vs. [the relu figure](src/plots/relu.pdf).
The loss function between is plotted for parameters "in between" the original neural net and another neural net with permuted hidden units. Let **L(s)** be the loss function, and *s* be the trained parameters of the original net. Let  **s'** be the parameter after permuting hidden units. We plotted **L(s t + s' (1-t))** for a bunch of random permutations vs. **t in [-1, 1]**.


