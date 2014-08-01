% convert a imsize(1) by imsize(2) by numchan image to a bunch of
% patches of block(1) by block(2) in increment of step
% the output is of size prod(block) by numblocks by sizebatch
% which can then be feed into LayerPatches
classdef LayerImage2Patch < LayerBase

    properties
        imsize
        block
        step
        numchan
        
        blockindices
        blockindiceschan
        flatblockindices
        numpatch
        dimpatch
    end

    methods
        function self = LayerImage2Patch(imsize, block, step, numchan, options)
            
            % handle channels here instead of indices to make things faster
            self.blockindices = uint32(getPatchIndex(imsize, block, step, 1));
            self.blockindiceschan = uint32(getPatchIndex(imsize, block, step, numchan));
            self.flatblockindices = self.blockindices(:);
            self.imsize = imsize; self.block = block; self.step = step; self.numchan = numchan;
            
            self.dimpatch = size(self.blockindices, 1);
            self.numpatch = size(self.blockindices, 2);
            
            self.name = ['MakePatches-' sprintf('%d-patches of size %d', self.numpatch, self.dimpatch) ];
        end
        
        function output=forward(self, input)
            self.input = input;
            [imgsizebynumchan, sizebatch] = size(input); %#ok<ASGLU>
            inputchanbybatch = reshape(input, prod(self.imsize), self.numchan*sizebatch);
            
            output = reshape(inputchanbybatch(self.flatblockindices, :), self.dimpatch, self.numpatch, self.numchan, sizebatch);
            output = permute(output, [1, 3, 2, 4]);
            output = reshape(output, self.dimpatch * self.numchan, self.numpatch, sizebatch);
        end
        
        % some of this can probably be made much faster
        function dLdin = backward(self, dLdout)
            [~, numpatch, sizebatch] = size(dLdout);
            sizevecimg = prod(self.imsize) * self.numchan;
            
            dLdin = zeros(sizevecimg, sizebatch); 
            for patchind = 1:numpatch
                dLdin(self.blockindiceschan(:, patchind), :) = ...
                    dLdin(self.blockindiceschan(:, patchind), :) + ...
                    reshape(dLdout(:, patchind, :), [], sizebatch);
            end
        end
    end
    
end