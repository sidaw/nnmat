classdef LayerPatches < LayerBase
    % write a description of the class here.
    properties
        numin
        numout
        numpatches
    end

    methods
        function L = LayerPatches(numin, numout, numpatches, options)
            L.params = convertType(0.01/sqrt(numin)*randn(numout, numin));
            L.name = ['LaterPatches' sprintf('%d-by-%d-by-%d', numin, numout, numpatches) ];
            L.numin = numin; L.numout = numout; L.numpatches = numpatches;
        end
        
        function output=forward(self, input)
            self.input = input;
            [patchdim, numpatch, sizebatch] = size(input);
            flatinput = reshape(input, patchdim, []);
            output = reshape(self.params * flatinput, self.numout, numpatch, sizebatch);
            
            self.output = output;
        end
        
        % some of this can probably be made much faster
        function dLdin = backward(self, dLdout)
            [outdim, numpatch, sizebatch] = size(dLdout);
            flatdLdout = reshape(dLdout, outdim, []);
            
            grad = convertType(zeros(self.numout, self.numin));
            
            dLdin = reshape(self.params' * flatdLdout, self.numin, numpatch, sizebatch);
            %dLdin = zeros(self.numin, numpatch, sizebatch);
            
            flatinput = reshape(self.input, self.numin, []);
            grad = flatdLdout * flatinput';
            
            self.grad = grad;
        end
    end
    
end