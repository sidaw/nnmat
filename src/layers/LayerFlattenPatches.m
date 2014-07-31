classdef LayerFlattenPatches < LayerBase
    % flattens a dimpatch by numpatch by sizebatch input to
    % an output of prod(dimpatch, numpatch) by sizebatch output
    % optionally tranposes first before flattening
    properties
        dimpatches
        numpatches
        numout
        transpose
    end

    methods
        function self = LayerFlattenPatches(dimpatches, numpatches, transpose)
            self.name = ['FlattenPatches' sprintf('-size-%d-amount-%d', dimpatches, numpatches) ];
            self.dimpatches = dimpatches; self.numpatches = numpatches;
            self.numout = dimpatches * numpatches;
            
            if ~exist('transpose', 'var')
                self.transpose = 0;
            else
                self.transpose = transpose;
            end
        end
        
        function output=forward(self, input)
            [outdim, numpatch, sizebatch] = size(input);
            self.input = input;
            
            if self.transpose == 1
                input = permute(input, [2,1,3]);
            end
            
            self.output = reshape(input, outdim * numpatch, sizebatch);
            output=self.output;
        end
        
        % check if this is the correct transpose
        function dLdin = backward(self, dLdout)
           
           if self.transpose == 0
               dLdin = reshape(dLdout, self.dimpatches, self.numpatches, []);
           else
               dLdin = reshape(dLdout, self.numpatches, self.dimpatches, []);
               dLdin = permute(dLdin, [2,1,3]);
           end
        end
    end
    
end