gpuArray = @(X) single(X);
W = gpuArray(randn(1000, 784));
data = gpuArray(randn(784,60000));
hids = single(zeros(1000, 60000));
batchsize = 100
numbatch = size(data,2)/batchsize;

tic
for i=1:numbatch
  current = (i-1)*batchsize+1:i*batchsize;
  hids(:,current) = max(0,W*data(:,current));
end
toc