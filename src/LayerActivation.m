classdef LayerActivation < LayerBase
 % the non-linearity in a network y_i = f(x_i)
properties
    actfunc
    gradfunc
end

methods
    function L = LayerActivation(numin, actname, options)
      L.name = ['Activation-' actname];
      [actfunc, gradfunc] = getActivation(actname);
      L.gradfunc = gradfunc;
      L.actfunc = actfunc;
      L.params = zeros(numin, 1);
    end

    function output=forward(self, input)
      self.input = input;
      self.output = self.actfunc( bsxfun(@plus, input, self.params) );
      output=self.output;
    end

    function dLdin = backward(self, dfdo)
      dLdin = self.gradfunc(self.input, self.output, dfdo);
      self.grad = sum(dLdin,2);
    end
end

end
