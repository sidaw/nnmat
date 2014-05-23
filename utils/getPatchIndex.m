function ttt = getPatchIndex(imsize, block, step)
    ma = imsize(1);
    na = imsize(2);
    m = block(1); n = block(2);
    
    if any([ma na] < [m n]) % if neighborhood is larger than image
        b = zeros(m*n,0);
        return
    end
    
    % Create Hankel-like indexing sub matrix.
    mc = block(1); nc = ma-m+1; nn = na-n+1;
    cidx = (0:mc-1)'; ridx = 1:step:nc;
    t = cidx(:,ones(length(ridx),1)) + ridx(ones(mc,1),:);    % Hankel Subscripts
    tt = zeros(mc*n,length(ridx));
    rows = 1:mc;
    for i=0:n-1,
        tt(i*mc+rows,:) = t+ma*i;
    end
    ttt = zeros(mc*n, length(ridx)*ceil(nn/step-1));
    cols = 1:length(ridx);
    for j=0:ceil(nn/step-1),
        ttt(:,j*length(ridx)+cols) = tt+ma*j*step;
    end
    
end