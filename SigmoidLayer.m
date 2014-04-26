function L = SigmoidLayer(numin, options)
    L.name = 'SigmoidLayer';
    L.params = [];
    L.forward = @(L, input) 1./(1+exp(-input));
    L.backward = @backprop;
    
    function [dLdin, dparams] = backprop(L, dLdout)
        dLdin = (L.output) .* (1-L.output) .* dLdout;
        dparams = [];
    end
end
