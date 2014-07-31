classdef LayerLinearPositive < LayerBase
    % basic linear layer, but only allowing positive weights.
    % usually does not work well, but good for analyzing optimization
    methods
        function L = LayerLinearPositive(numin, numout, options)
            L.name = 'Linear';
            L.params = convertType( 0.01/sqrt(numin)*abs(randn(numout, numin)) );
        end
        
        function output=forward(self, input)
            self.input = input;
            self.output = self.params*input;
            output=self.output;
        end
        
        function dLdin = backward(self, dLdout)
            self.grad = dLdout * self.input';
            dLdin = self.params' * dLdout;
        end
        
         function setparams(self, X)
             X = max(X,0);
             self.params = reshape(X, size(self.params));
         end
    
    end
    
end
