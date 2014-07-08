classdef LayerNoising < LayerBase
    % the non-linearity in a network y_i = f(x_i)
    properties
        actfunc
        gradfunc
        state
        droprate
        
        testing
        fixedmask
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
                if self.fixedmask
                    rng(1)
                end
                self.state = 1*(rand(size(input))>self.droprate);
                self.output = self.state.*input;
            else
                self.output = (1-self.droprate)*input;
            end
            output=self.output;
        end
        
        function dLdin = backward(self, dfdo)
            dLdin = dfdo .* self.state;
            self.grad = [];
        end
    end
    
end
