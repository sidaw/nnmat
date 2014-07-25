datapath = '~/data/cifar10/cifar-10-batches-mat/';

X = zeros(3072, 50000, 'single');
y = ones(10, 50000);

for i = 1:5
    dataname = sprintf('data_batch_%d.mat', i);
    load([datapath dataname])
    startind = (i-1)*10000+1;
    endind = i*10000;
    X(:, startind:endind) = single(data')/255;
    y(:,startind:endind) = to1ofk(single(labels), 10)';
end

load([datapath 'test_batch.mat'])
Xtest = single(data')/255;
ytest = to1ofk(single(labels), 10)';

save([datapath 'cifarall.mat'], 'X', 'y', 'Xtest', 'ytest');
