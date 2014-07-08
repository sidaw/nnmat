addpath(genpath('utils'))

datapath = '~/data/mnisty/affnist_mat.mat';
load(datapath)

dimdata =  batchdata;
numhid = 250;
numclass = 10;
numdata = 10;

L = {};
L{1} = LinearLayer(dimdata, numhid);
L{2} = ActivationLayer(numhid, 'sigmoid');
L{3} = LinearLayer(numhid, numclass);
nn = Sequential(L{:});

data = randn(dimdata, numdata);
y = randn(numclass, numdata);

params = nn.getparams();

for iter = 1:200
    iter
    output = nn.forward(data);
    losses(iter) = sum( (output(:) - y(:)).^2 );
    %[output; y]
    dLdout = output - y;

    dldin = nn.backward(dLdout);
    grad = nn.getgrad();
    params = params - 0.01* grad;
    
    nn.setparams(params)
    b=nn.layers{2}.getparams();
end