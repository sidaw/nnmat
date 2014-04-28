classdef Module
 % write a description of the class here.
properties
     input
     output
     params
end

methods
    function output = forward(self, input)
      output = input;
    end

    function grad = backward(self, dfdo)
      grad = dfdo;
    end

    % convert parameters describing this module to a vector
    function [X] = getparams(self)
      X = self.params;
    end

    function setparams(self, X)
      self.params = X;
    end
end

end
