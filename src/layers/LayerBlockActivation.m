classdef LayerBlockActivation < LayerBase
 % the non-linearity in a network y_i = f(x_i)
properties
    actfunc
    gradfunc
    hasbias
    
    numblocks
    numin
    blocksize
    
    output_reshape
    input_reshape
end

% the block activation function is supposed to act on a single block,
% extending it to a whole layer of many blocks is handled in this function
methods
    function L = LayerBlockActivation(numin, actname, blocksize, options)
      L.name = ['BlockActivation-' actname];
      [actfunc, gradfunc] = getBlockActivation(actname);
      L.gradfunc = gradfunc;
      L.actfunc = actfunc;
      
      assert(mod(numin, blocksize) == 0, 'blocksize does not divide evenly into numin')
      L.numin = numin; L.blocksize = blocksize;
      L.numblocks = numin / blocksize; 
      
      if ~exist('options', 'var')
        options.exist = 1; % cheat to initialize the options
      end
      L.hasbias = parseOption(options, 'hasBias', 1);
      
      L.params = convertType(zeros(numin, 1));
    end

    function output = forward(self, input)
      inputandbias = bsxfun(@plus, input, self.params);
      self.input_reshape = reshape(inputandbias, self.blocksize, []);
      
      output_reshape = self.actfunc(self.input_reshape);
      self.output_reshape = output_reshape;
      
      output = reshape(output_reshape, self.numin, []);
    end

    function dLdin = backward(self, dfdo)
      dfdo_reshape = reshape(dfdo, self.blocksize, []);
      dLdin_reshape = self.gradfunc(self.input_reshape, self.output_reshape, dfdo_reshape);
      
      dLdin = reshape(dLdin_reshape, self.numin, []);
      
      if self.hasbias
        if length(size(dfdo)) == 2
            self.grad = sum(dLdin,2);
        end
      else
        self.grad = zeros(size(self.params));
      end
    end
end

end
