classdef LayerNoising < LayerBase
 % the non-linearity in a network y_i = f(x_i)
properties
    actfunc
    gradfunc
    state
    droprate
    testing
end

methods
    function L = LayerNoising(droprate)
      L.name = ['Noising'];
      L.droprate = droprate;
      L.params = [];
      L.testing = 0;
    end

    function output=forward(self, input)
      self.input = input;
      if ~self.testing
	self.state = rand(size(input))>self.droprate;
	self.output = self.state.*input;
	output=self.output;
      else	
	self.output = self.droprate*input;
	output=self.output;      
      end
    end

    function dLdin = backward(self, dfdo)
      dLdin = dfdo .* self.state;
      self.grad = [];
    end
end

end
