classdef LinearLayer < Module
 % write a description of the class here.
properties
     input
     output
     params
end

methods
    function L = LinearLayer(numin, numout, options)
      L.name = 'LinearLayer';
      L.params = 0.01/sqrt(numin)*randn(numout, numin);
      L.backward = @backprop;
    end

    function output=forward(self, input)
      self.output = L.params*input;
      output=self.output
    end

    function backward(self, dfdo)
      self.dLdin = L.params' * dLdout;
      self.grad = dLdout * L.input';
    end

    % convert parameters describing this module to a vector
    % so we can have one optimization interface
    function [X] = getparams(self)
      X = reshape(self.params, [], 1);
    end

    function setparams(self, X)
      self.params = reshape(X, );
    end
end

end
