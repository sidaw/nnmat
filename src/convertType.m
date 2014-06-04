function typeddata = convertType(x)
if ~testToolboxes('Parallel Computing Toolbox')
    typeddata = single(x);
else
    typeddata = gpuArray(single(x));
end