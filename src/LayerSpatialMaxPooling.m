classdef LayerSpatialMaxPooling < LayerBase
    % write a description of the class here.
    properties
        numin
        numout
        numpatches
        numchannels
        sizevectimg;
        
        blockindices
    end

    methods
        function self = LayerPooling(sizeimg, sizeregion, sizestride)
            self.name = ['LayerPooling' sprintf('%d-by-%d img %d-by-%d regions strides of %d',...,
                sizeimg(1), sizeimg(2), sizeregion(1), sizeregion(2), sizestride) ];
            self.blockindices = uint32(getPatchIndex(sizeimg, sizeregion, sizestride));
        end
        
        function output=forward(self, input)
            % input should be of size prod(sizeimg) by number of channels,
            % by sizebatch
            [numchannels, sizevectimg, sizebatch] = size(input); %#ok<*PROP>
            self.numchannels = numchannels;
            self.sizevectimg = sizevectimg;
            
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
            
            for i=1:numregions
                currentargmax = self.argmaxs(:,i,:);
                inds = sub2ind(currentargmax, [numchannels, numregions, sizebatch]);
                dLdin(c, self.blockindices(currentargmax(c, d) ,i), d) = dLdin(inds) + dLdout(:,i,:);
            end
        end
    end
    
end