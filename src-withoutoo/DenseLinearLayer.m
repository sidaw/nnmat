function L = DenseLinearLayer(numin, numout, options)
    L.name = 'DenseLinearLayer';
    L.params = 0.01/sqrt(numin)*randn(numout, numin);
    L.forward = @(L, input) L.params*input;
    L.backward = @backprop;
    
    function [dLdin, dparams] = backprop(L, dLdout)
        dLdin = L.params' * dLdout;
        dparams = dLdout * L.input';
    end
end