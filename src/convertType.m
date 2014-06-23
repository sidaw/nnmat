function typeddata = convertType(x)
if ~testToolboxes('Parallel Computing Toolbox')
    typeddata = double(x);
else
    typeddata = gpuArray(single(x));
end