classdef LinearLayer < Module
 % write a description of the class here.
methods
    function L = LinearLayer(numin, numout, options)
      L.name = 'LinearLayer';
      L.params = 0.01/sqrt(numin)*randn(numout, numin);
    end

    function output=forward(self, input)
      self.input = input;
      self.output = self.params*input;
      output=self.output;
    end

    function dLdin = backward(self, dLdout)
      self.grad = dLdout * self.input';
      dLdin = self.params' * dLdout;
    end

    % convert parameters describing this module to a vector
    % so we can have one optimization interface
    function [X] = getparams(self)
      X = self.params(:);
    end

    function setparams(self, X)
      self.params(:) = X;
    end

    function [grad] = getgrad(self)
      grad = self.grad(:);
    end

    
end

end
