function typeddata = convertType(x)
if 1 %~testToolboxes('Parallel Computing Toolbox')
    typeddata = double(x);
else
    typeddata = gpuArray(single(x));
end