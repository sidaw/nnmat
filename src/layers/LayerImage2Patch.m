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
        reverseindices
        numpatch
        dimpatch
    end
    
    methods
        function self = LayerImage2Patch(imsize, block, step, numchan, options) %#ok<INUSD>
            
            self.imsize = imsize; self.block = block; self.step = step; self.numchan = numchan;
            
            % handle channels here instead of indices to make things faster
            self.blockindices = uint32(getPatchIndex(imsize, block, step, 1));
            self.blockindiceschan = uint32(getPatchIndex(imsize, block, step, numchan));
            self.flatblockindices = self.blockindices(:);
            self.reverseindices = LayerImage2Patch.getReverseIndices(self.flatblockindices, self.imsize);
            
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
        
        
        function dLdin = backward(self, dLdout)
            if 0
                [dimpatchtimeschan, numpatch, sizebatch] = size(dLdout); %#ok<PROP,ASGLU>
                reshape_dLdout = reshape(dLdout, self.dimpatch, self.numchan, ...
                    self.numpatch, sizebatch);
                
                % so that patches of channel 1 are all adjacent, then
                % channel 2 and so on
                permute_dLdout = permute(reshape_dLdout, [1, 3, 2, 4]);
                
                flatdLdout = zeros(self.dimpatch*self.numpatch+1, self.numchan, sizebatch);
                flatdLdout(1:self.dimpatch*self.numpatch,:, :) = reshape(permute_dLdout, ...
                    self.dimpatch*self.numpatch, self.numchan, sizebatch);
                
                
                maxpatches = size(self.reverseindices, 1);
                flatreverseindices = self.reverseindices(:);
                
                flatdLdin = flatdLdout(flatreverseindices,:,:);
                dLdinbeforesum = reshape(flatdLdin, ...
                    maxpatches, prod(self.imsize) * self.numchan, sizebatch);
                
                dLdin = squeeze(sum(dLdinbeforesum, 1));
            else
                [dimpatchtimeschan, numpatch, sizebatch] = size(dLdout); %#ok<PROP,ASGLU>
                sizevecimg = prod(self.imsize) * self.numchan;
                dLdin = zeros(sizevecimg, sizebatch);
                for patchind = 1:self.numpatch
                    dLdin(self.blockindiceschan(:, patchind), :) = ...
                        dLdin(self.blockindiceschan(:, patchind), :) + ...
                        reshape(dLdout(:, patchind, :), [], sizebatch);
                end
            end
        end
        % given a flattned index that transform the image into patches, get
        % the reverse index that allows for efficient back propagation
        % implementation.
        % fbi: the flatblockindices
            % imsize: size of the image
            % this assumes that each pixel appears in roughly the same number of patches,
            % could be bad if some pixel appears in a vast number of patches
    end
    
    methods(Static)
        function reverseindices = getReverseIndices(fbi, imsize)
            % the maximum number of patches that any pixel appears in
            maxpatches = sum(mode(fbi)==fbi);
            imlength =  prod(imsize);
            
%             if max(fbi) ~= imlength
%                 warn('some pixels are never used in patches')
%             end
            
            % initialize the reverse index to point to 1+imlength
            reverseindices = imlength + 1 + uint32(zeros(maxpatches, imlength));
            [sortedfbi, indfbi] = sort(fbi, 'ascend');
            
            % populating the reverseindices
            prevind = 0;
            for i = 1:length(fbi)
                curind = sortedfbi(i);
                if curind ~= prevind
                    curpos = 1;
                    prevind = curind;
                end
                
                reverseindices(curpos, curind) = indfbi(i);
                curpos = curpos + 1;
            end
        end
    end
end