classdef LayerNoising < LayerBase
 % the non-linearity in a network y_i = f(x_i)
properties
    actfunc
    gradfunc
    state
    droprate
end

methods
    function L = LayerNoising(droprate)
      L.name = ['Noising'];
      L.droprate = droprate;
      L.params = [];
    end

    function output=forward(self, input)
      self.input = input;
      self.state = rand(size(input))>self.droprate;
      self.output = self.state.*input;
      output=self.output;
    end

    function dLdin = backward(self, dfdo)
      dLdin = dfdo .* self.state;
      self.grad = [];
    end
end

end
