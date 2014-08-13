% Sample subimages randomly from a larger image
classdef LayerSampleImage < LayerBase
    
    properties
        subsize
        imsize
        samplelimit
        numchan
        
        baseindices
    end
    
    methods

        function self = LayerSampleImage(imsize, subsize, numchan, options) %#ok<INUSD>
            self.subsize = subsize(:)'; self.numchan = numchan;
            self.imsize = imsize(:)';
            self.samplelimit = self.imsize - self.subsize;
            
            self.name = ['SampleImage-' sprintf('sample-%s-on-imagesize-%s', ...
                                             self.subsize, self.imsize)];
            gridr = (1:self.subsize(1));
            gridc = (1:self.subsize(2));
            [sub2, sub1] = meshgrid(gridc, gridr); 
            self.baseindices = [sub1(:), sub2(:)];
            
        end
        
        function output=forward(self, input)
            self.input = input;
            [imsizebynumchan, sizebatch] = size(input); %#ok<ASGLU>
            
            output = zeros(prod(self.subsize) * self.numchan, sizebatch);

            for i = 1:sizebatch
                imagei = input(:, i);
                offset = floor(rand(1,2).* (self.samplelimit+1));
                indices = bsxfun(@plus, self.baseindices, offset(:)');
                linearindices = sub2ind(self.imsize, indices(:, 1), indices(:,2) );
                inputchans = reshape(imagei, prod(self.imsize), self.numchan);
                outputchans = inputchans(linearindices, :);
                output(:, i) = outputchans(:);
            end
        end
        
        
        function dLdin = backward(self, dLdout)
        % Not implemented, this layer is expected only to be applied to raw images
            dLdin = 0;
        end        
end
