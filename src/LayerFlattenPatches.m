classdef LayerFlattenPatches < LayerBase
    % write a description of the class here.
    properties
        dimpatches
        numpatches
        dimout
    end

    methods
        function L = LayerFlattenPatches(dimpatches, numpatches, options)
            L.name = ['FlattenPatches' sprintf('-size-%d-amount-%d', dimpatches, numpatches) ];
            L.dimpatches = dimpatches; L.numpatches = numpatches;
            L.dimout = dimpatches * numpatches;
        end
        
        function output=forward(self, input)
            [outdim, numpatch, sizebatch] = size(input);
            self.input = input;
            self.output = reshape(input, outdim * numpatch, sizebatch);
            output=self.output;
        end
        
        function dLdin = backward(self, dLdout)
           dLdin = reshape(dLdout, self.dimpatches, self.numpatches, []);
        end
    end
    
end