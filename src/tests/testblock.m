A = [1, 3.14, 2.71; -1 0 6.28];
nrow = 50;
ncol = 60;
selrow = 100;
selcol = 30;

oneblock = ones(nrow,ncol);

data = randn(selrow*nrow, selcol*ncol);

sel = randi(selcol,selrow,1);
inds = (1:size(sel,1))';
values = ones(size(sel));

tic;
selmat = zeros(selrow,selcol);
indsel = sub2ind(size(selmat), inds, sel);
selmat(indsel) = 1;
realselmat = kron(selmat, oneblock)>0;
datatrans = data';
selected = datatrans(realselmat');
selected = reshape(selected, ncol, size(sel,1)*nrow)';
toc;

tic;
resdata = reshape(data, nrow*selrow*selcol, ncol);

toc;


tic
selectedloop = zeros(selrow*nrow, ncol);
for i=1:length(sel)
    indstart = (i-1)*nrow + 1;
    indend = indstart + nrow - 1;
    
    colstart = (sel(i)-1)*ncol + 1;
    colend = (sel(i))*ncol;
    
    selectedloop(indstart:indend, :) = data(indstart:indend, colstart:colend);
    
end
toc;

shouldbe0 = any(~(selected(:)==selectedloop(:)))