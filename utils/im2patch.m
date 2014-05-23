function Xpatch=im2patch(X, imsize, block, step)
    ttt = getPatchIndex(imsize, block, step);
    Xpatch = zeros([size(ttt), size(X,2)]);
    for i=1:size(X,2)
        Xcurrent = X(:,i);
        Xpatch(:,:,i) = Xcurrent(ttt);
    end
end
