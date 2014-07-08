classdef LayerImage2Patch < LayerBase
    % initial deterministic data processing layer, does not support
    % backprop through it
    properties
        imsize
        block
        step
        
        blockindices
        numpatches
        dimpatches
    end

    methods
        function self = LayerImage2Patch(imsize, block, step, options)
            self.blockindices = uint32(getPatchIndex(imsize, block, step));
            self.imsize = imsize; self.block = block; self.step = step;
            [self.dimpatches, self.numpatches] = size(self.blockindices);
            self.name = ['MakePatches-' sprintf('%d-patches of size %d', self.numpatches, self.dimpatches) ];
        end
        
        function output=forward(self, input)
            self.input = input;
            self.output = convertType(zeros([size(self.blockindices), size(input,2)]));
            for i=1:size(input,2)
                Xcurrent = input(:,i);
                self.output(:,:,i) = Xcurrent(self.blockindices);
            end
            output=self.output;
        end
        
        % some of this can probably be made much faster
        function dLdin = backward(self, dLdout)
            dLdin = 0;
        end
    end
    
end