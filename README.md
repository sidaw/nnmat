# nnmat

Just an experimental deep learning library in object oriented matlab.
To run a 2 hidden layers network with dropout on mnist, you simply needs the following:

'''matlab
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
'''

The data should be in a matrix of datadim by numhid. See test/testmnist for the whole thing.
nnmat more or less followed the torch/nn design, and can run on gpu if set up correctly.

## how to run

go to scr

initnnmat
testconvextrain

or look at any of these testX.m

## sigmoid vs. relu

It appears that relu and sigmoid units can make a big difference. This is well-known but poorly understood.
Try:

initnnmat

testconvextrain; plotObjectivePermute

Then comment out the sigmoid layer and uncomment the relu line.

'''matlab
L{end+1} = LayerActivation(numhid, 'sigmoid');
%L{end+1} = LayerActivation(numhid, 'relu');
'''

Run this again: testconvextrain; plotObjectivePermute

I got the [sigmoid figure](src/plots/sigmoid.pdf) vs. [the relu figure](src/plots/relu.pdf), and interested in an explaination why they are so different.

The plot trace the objective function between the original network and a network with permuted hidden units.


