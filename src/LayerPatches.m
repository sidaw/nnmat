classdef LayerPatches < LayerBase
    % write a description of the class here.
    properties
        numin
        numout
        numpatches
    end

    methods
        function L = LayerPatches(numin, numout, numpatches, options)
            L.params = 0.01/sqrt(numin)*randn(numout, numin);
            L.name = ['LaterPatches' sprintf('%d-by-%d-by-%d', numin, numout, numpatches) ];
            L.numin = numin; L.numout = numout; L.numpatches = numpatches;
        end
        
        function output=forward(self, input)
            self.input = input;
            [patchdim, numpatch, sizebatch] = size(input);
            output = zeros(self.numout, numpatch, sizebatch);
            
            for i = 1:sizebatch
               output(:,:, i) = self.params * input(:,:,i);
            end
            
            self.output = output;
        end
        
        % some of this can probably be made much faster
        function dLdin = backward(self, dLdout)
            [outdim, numpatch, sizebatch] = size(dLdout);
            
            grad = zeros(self.numout, self.numin);
            dLdin = zeros(self.numin, numpatch, sizebatch);
            for i = 1:sizebatch
               grad = grad + dLdout(:,:,i) * self.input(:,:,i)';
               dLdin(:,:,i) = self.params' * dLdout(:,:,i);
            end
            self.grad = grad;
        end
    end
    
end