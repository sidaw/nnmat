classdef LayerLinear < LayerBase
    % write a description of the class here.
    methods
        function L = LayerLinear(numin, numout, options)
            L.name = 'Linear';
            L.params = convertType( 0.01/sqrt(numin)*randn(numout, numin) );
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
    end
    
end
