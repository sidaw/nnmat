classdef LayerGenerateFakeData < LayerBase
 % just feed the parameter as data to the next layer, a pseudo layer for
 % testing.
properties
    sizedata
end

methods
    function self = LayerGenerateFakeData(sizedata)
        self.params = randn(sizedata);
    end

    function output = forward(self, input)
      output = self.params;
    end

    function dLdin = backward(self, dfdo)
      self.grad = dfdo;
      dLdin = 0;
    end
end

end
