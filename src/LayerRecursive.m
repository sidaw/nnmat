classdef LayerRecursive < LayerBase
 % write a description of the class here.
properties
    inputs
    output
    grads
    
    iters
    layer
end

methods
    function self = LayerRecursive(layer, iters)
      self.inputs = {};
      self.outputs = {};
      if iters == 1
          warning('why bother using recursive layer if you are only doing it once?');
      end
      self.layer = layer;
      self.iters = iters;
    end

    function output=forward(self, input)
      self.inputs{1} = input;
      for i = 2:self.iters
        input = self.layer.forward(input);
        self.inputs{i} = input;
      end
      output = self.layer.forward(input);
      self.output = output;
    end

    function dLdin = backward(self, dfdo)
      dLdin = dfdo;
      for i = self.iters:-1:1
        self.layer.input = self.inputs{i};
        dLdin = self.layer.backward(dLdin);
        self.grads{i} = self.layer.getgrad();
      end
    end

    % just set the parameter to the internal layer
    function [X] = getparams(self)
      X = self.layer.getparams();
    end

    function setparams(self, X)
      self.layer.setparams(X);
    end

    function [grad] = getgrad(self)
      grad = self.grads{1};
      for i = 2:self.iter
          grad = grad + self.grads{i};
      end
    end

end

end
