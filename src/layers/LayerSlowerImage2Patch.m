% convert a imsize(1) by imsize(2) by numchan image to a bunch of
% patches of block(1) by block(2) in increment of step
% the output is of size prod(block) by numblocks by sizebatch
% which can then be feed into LayerPatches
classdef LayerSlowerImage2Patch < LayerBase

    properties
        imsize
        block
        step
        numchan
        
        blockindices
        numpatch
        dimpatch
    end

    methods
        function self = LayerSlowerImage2Patch(imsize, block, step, numchan, options)
            self.blockindices = uint32(getPatchIndex(imsize, block, step, numchan));
            
            self.imsize = imsize; self.block = block; self.step = step; self.numchan = numchan;
            
            [self.dimpatch, self.numpatch] = size(self.blockindices);
            
            self.name = ['MakePatches-' sprintf('%d-patches of size %d', self.numpatch, self.dimpatch) ];
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
            [~, numpatch, sizebatch] = size(dLdout);
            sizevecimg = prod(self.imsize) * self.numchan;
            
            dLdin = zeros(sizevecimg, sizebatch); 
            for patchind = 1:numpatch
                dLdin(self.blockindices(:, patchind), :) = ...
                    dLdin(self.blockindices(:, patchind), :) + ...
                    reshape(dLdout(:, patchind, :), [], sizebatch);
            end
        end
    end
    
end