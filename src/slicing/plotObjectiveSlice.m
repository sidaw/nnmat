
close all

hold on

for j = 1:5
randdir = randn(size(params));

randdir = zeros(size(params));
firstlayer = 784*200;
laterlayer = 784*200+200;

randind = randi(length(params),1,1);
%randind = randi(1000,1,1) + laterlayer;
randdir(randind) = 1;

%randdir = randn(size(paramsopt));

randdir = randdir / norm(randdir) * norm(paramsopt);
steps = 0.01*[-1:0.01:1];
loss = [];
lossopt = [];
iter = 0;
for s = steps
    iter = iter +  1;
    if mod(iter,50) == 0
        disp(iter)
    end
    paramstest = params + s*randdir;
    loss(end+1) = batchlossfunc(paramstest);
    
    paramstestopt = paramsopt + s*randdir;
    lossopt(end+1) = batchlossfunc(paramstestopt);
end
hold on
figure(1)
plot(steps, loss, 'b')

hold on
figure(2)
plot(steps, lossopt, 'r');
end

hold off