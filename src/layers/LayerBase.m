classdef LayerBase < handle
 % write a description of the class here.
properties
     input
     output
% these two should store the parameters and their corresponding gradient in whatever format that is most conveniently used
% whereas the functions for getting and setting params and grad should always convert these to vectors so that we have a uniform optimization routine
     params
     grad
     name
end

methods
    function output = forward(self, input)
      self.output = input;
      output = self.output;
    end

    function grad = backward(self, dfdo)
      self.grad = dfdo;
      grad = self.grad;
    end

    % convert parameters describing this module to a vector
    % so we can have one optimization interface
    function [X] = getparams(self)
      X = self.params(:);
    end

    function setparams(self, X)
      self.params = reshape(X, size(self.params));
    end

    function [grad] = getgrad(self)
      grad = self.grad(:);
    end
end

end
