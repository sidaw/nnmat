classdef LayerActivation < LayerBase
 % the non-linearity in a network y_i = f(x_i)
properties
    actfunc
    gradfunc
    hasbias
end

methods
    function L = LayerActivation(numin, actname, options)
      L.name = ['Activation-' actname];
      [actfunc, gradfunc] = getActivation(actname);
      L.gradfunc = gradfunc;
      L.actfunc = actfunc;
      
      if exist('options', 'var')
        L.hasbias = parseOption(options, 'hasBias', 1);
      end
      
      L.params = zeros(numin, 1);
    end

    function output = forward(self, input)
      self.input = input;
      self.output = self.actfunc( bsxfun(@plus, input, self.params) );
      output = self.output;
    end

    function dLdin = backward(self, dfdo)
      dLdin = self.gradfunc(self.input, self.output, dfdo);
      if self.hasbias
        if length(size(dfdo)) == 2
            self.grad = sum(dLdin,2);
        elseif length(size(dfdo)) == 3
            self.grad = sum(sum(dLdin,2),3);
        end
      else
        self.grad = zeros(size(self.params));
      end
    end
end

end
