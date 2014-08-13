classdef LayerLinear < LayerBase
    % write a description of the class here.
    methods
        function L = LayerLinear(numin, numout, options)
            if ~exist('options', 'var'); options.empty = 1; end
            
            initW = parseOption(options, 'initW', 1e-3);
            
            L.name = 'Linear';
            
            L.params = convertType( initW * randn(numout, numin) );
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
