function typeddata = convertType(x)
if 1 %~testToolboxes('Parallel Computing Toolbox')
    typeddata = single(x);
else
    typeddata = gpuArray(single(x));
end