classdef LayersFork < LayerBase
 % apply each of the containing layer to the same input, producing a
 % concatenated ouput
 % all layers should take the same input
properties
    layers
    decodeinfo
end

methods
    function L = LayersFork(varargin)
      if length(varargin) == 1
          warning('No need to use LayersFork if it only contains one element')
          if iscell(varagin{1})
              warning('attempting to use first item as the list')
              varargin = varagin{1};
          end
      end
      
      L.layers = varargin;
    end

    function output=forward(self, input)
      output = input;
      for i = 1:length(self.layers)
        output = self.layers{i}.forward(output);
      end
      self.output = output; self.input = input;
    end

    function dLdin = backward(self, dfdo)
      dLdin = dfdo;
      for i = length(self.layers):-1:1
        dLdin = self.layers{i}.backward(dLdin);
      end
    end

    % convert parameters describing this module to a vector
    % so we can have one optimization interface
    function [X] = getparams(self)
      X = [];
      self.decodeinfo = [];
      for i = 1:length(self.layers)
        X = [X; self.layers{i}.getparams()];
        self.decodeinfo = [self.decodeinfo; length(X)];
      end
    end

    function setparams(self, X)
      start = 1;
      for i = 1:length(self.layers)
        if self.decodeinfo(i) >= start
            self.layers{i}.setparams(X( start:self.decodeinfo(i) ));
            start = self.decodeinfo(i)+1;
        end
      end
    end

    function [grad] = getgrad(self)
      grad = [];
      for i = 1:length(self.layers)
        grad = [grad; self.layers{i}.getgrad()];
      end
    end

end

end
