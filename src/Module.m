classdef Module < handle
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

    function grad=getgrad(self)
      grad = [];
    end

end

end
