classdef LayerSpatialMaxPooling < LayerBase
    % write a description of the class here.
    properties
        numin
        numout
        numpatches
        numchannels
        numregions
        sizevectimg
        sizeimg
        
        blockindices
        blocksize
        
        argmaxs
    end

    methods
        % pool sizeregion by sizeregion inputs from the patches layer, with
        % a stride size of sizestride
        function self = LayerSpatialMaxPooling(sizeimg, sizeregion, sizestride)
            self.name = ['LayerPooling' sprintf('%d-by-%d img %d-by-%d regions strides of %d',...,
                sizeimg(1), sizeimg(2), sizeregion(1), sizeregion(2), sizestride) ];
            self.blockindices = uint32(getPatchIndex(sizeimg, sizeregion, sizestride));
            self.numpatches = size(self.blockindices, 2);
            self.blocksize = size(self.blockindices, 1);
            self.sizeimg = sizeimg;
            
        end
        
        function output=forward(self, input)
            % input should be of size prod(sizeimg) by number of channels,
            % by sizebatch
            [numchannels, sizevectimg, sizebatch] = size(input); %#ok<*PROP>
            self.numchannels = numchannels;
            self.sizevectimg = sizevectimg;
            assert(prod(self.sizeimg) == self.sizevectimg);
            
            [sizevectregion, numregions] = size(self.blockindices); %#ok<ASGLU>
            % output should be of size numchannels by numregions, by sizebatch 
            
            self.output = zeros(numchannels, numregions, sizebatch);
            
            self.input = input;
            self.argmaxs = zeros(numchannels, numregions, sizebatch);
            
            for i=1:numregions
                [currentpool, currentargmax] = max(input(:,self.blockindices(:,i),:), [], 2);
                self.output(:,i,:) = currentpool;
                self.argmaxs(:,i,:) = currentargmax;
            end
            output=self.output;
            
        end
        
        % some of this can probably be made much faster
        function dLdin = backward(self, dLdout)
            [numchannels, numregions, sizebatch] = size(dLdout); %#ok<ASGLU>
            dLdin = zeros(self.numchannels, self.sizevectimg, sizebatch);
            onetosizebatch = reshape(repmat(1:sizebatch, [numchannels, 1]), 1, []);
            channelindices = repmat(1:numchannels, [1, sizebatch]);
            
            
            for i=1:numregions
                currentargmax = reshape(self.argmaxs(:,i,:),1, []);
                currentblock = self.blockindices(:,i);
                
                sparsegradind = sub2ind([numchannels, size(currentblock,1), sizebatch], channelindices, currentargmax, onetosizebatch);
                updates = zeros(self.numchannels, self.blocksize, sizebatch);
                updates(sparsegradind) = reshape(dLdout(:, i, :), 1, []);
                dLdin(:, currentblock, :) = dLdin(:, currentblock ,:) + updates;
            end
        end
    end
    
end