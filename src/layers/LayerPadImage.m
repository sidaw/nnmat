% pad image by the specified amount on each side.
classdef LayerPadImage < LayerBase
    
    properties
        imsize
        padsize
        totalsize
        numchan
        
        centerindices
    end
    
    methods

        function self = LayerPadImage(imsize, padsize, numchan, options) %#ok<INUSD>
            self.imsize = imsize; self.numchan = numchan; self.padsize = padsize;

            % pad on each side of the 2D image by default
            self.totalsize = self.imsize + 2*self.padsize;

            self.name = ['PadImage-' sprintf('padd-%s-on-imagesize-%s', ...
                                             self.padsize, self.imsize)];
            gridr = self.padsize(1) + (1:self.imsize(1));
            gridc = self.padsize(2) + (1:self.imsize(2));
            [sub2, sub1] = meshgrid(gridc, gridr); 
            self.centerindices = sub2ind( self.totalsize, sub1(:), sub2(:) );
            
        end
        
        function output=forward(self, input)
            self.input = input;
            [imgsizebynumchan, sizebatch] = size(input); %#ok<ASGLU>
            inputflatchan = reshape(input, prod(self.imsize), self.numchan*sizebatch);
            output = zeros(prod(self.totalsize), self.numchan*sizebatch);
            output(self.centerindices, :) = inputflatchan;
            output = reshape(output, prod(self.totalsize)*self.numchan, sizebatch);
        end
        
        
        function dLdin = backward(self, dLdout)
            [sizebynumchan, sizebatch] = size(dLdout); %#ok<ASGLU>
            
            dLdoutflatchan = reshape(dLdout, prod(self.totalsize), self.numchan, sizebatch);
            dLdin = dLdoutflatchan(self.centerindices, :, :);
            dLdin = reshape(dLdin, prod(self.imsize)*self.numchan, sizebatch);
        end
    end
end
