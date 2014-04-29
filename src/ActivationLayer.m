classdef ActivationLayer < Module
 % write a description of the class here.
properties
    actfunc
    actgrad
end

methods
    function L = ActivationLayer(numin, actname, options)
    

      L.name = ['ActivationLayer-' actname];
      L.actfunc = actfuncs(actname);
      L.actgrad = actgrads(actname);
      L.params = zeros(numin, 1);
    end

    function output=forward(self, input)
      self.input = input;
      self.output = self.actfunc( bsxfun(@plus, input, self.params) );
      output=self.output;
    end

    function dLdin = backward(self, dfdo)
      dLdin = dfdo .* self.actgrad(self.input, self.output);
      self.grad = sum(dLdin,2);
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
