close all
testsize = 50;
batchlossfunc = @(params) BatchLossFunction_DivideData(params, X(:,1:testsize), y(:, 1:testsize), nn, 'nll_logprob', optsloss);
numhid = size(nn.layers{1}.params,1);

for j = 1:10
    steps = [-1:0.01:2];
    loss = [];
    lossopt = [];
    iter = 0;
    paramsold = paramsopt;
   % paramsold = params;
    permind = randperm(numhid);
%     permind = 1:numhid;
%     
%     ind1 = randi(numhid,1,1);
%     ind2 = randi(numhid,1,1);
%     permind(ind1) = ind2;
%     permind(ind2) = ind1;
    
    nn.setparams(paramsold);
    nn.layers{1}.params = nn.layers{1}.params(permind, :);
    nn.layers{2}.params = nn.layers{2}.params(permind);
    nn.layers{3}.params = nn.layers{3}.params(:, permind);
    paramsnew = nn.getparams();
    
    
    for s = steps
        iter = iter +  1;
        if mod(iter,50) == 0
            disp(iter)
        end
        paramstest = s*paramsold + (1-s)*paramsnew;
        loss(end+1) = batchlossfunc(paramstest);

   
    end
    figure(1)
    hold on
    plot(steps, loss, 'b')
end

hold off