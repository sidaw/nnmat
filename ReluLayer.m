function L = ReluLayer(numin, options)
    L.name = 'ReluLayer';
    L.params = [];
    L.forward = @(L, input) max(input,0);
    L.backward = @backprop;
    
    function [dLdin, dparams] = backprop(L, dLdout)
        dLdin = (L.input > 0).* dLdout;
        dparams = [];
    end
end
