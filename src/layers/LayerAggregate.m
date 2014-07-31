classdef LayerAggregate < LayerBase
    % takes an input of prod(patchsize, numchan) by numpatch by sizebatch
    % and outputs numchan by numpatch by sizebatch by somehow aggregating
    % the patches
    properties
        sizepatch % sizepatch actually means length of patch
        numchan
        numpatch
        backpropmask
        
        aggname
    end

    methods
        function self = LayerAggregate(sizepatch, numchan, numpatch, aggname)
            self.sizepatch = sizepatch;
            self.numchan = numchan; self.numpatch = numpatch;
            self.aggname = aggname;
        end
        
        % outputs a prod([numpatch,numchan]) by sizebatch array, after aggregating
        % each patch
        function output=forward(self, input)
            self.input = input;
            [sizepatchbynumchan, numpatch, sizebatch] = size(input); %#ok<PROP,ASGLU>
            input_reshape = reshape(input, self.sizepatch, self.numchan, self.numpatch, sizebatch);
            
            % argmax is numchan by numpatch by sizebatch
            
            
            switch self.aggname
                case 'max' % max pooling
                    [output] = max(input_reshape, [], 1);
                    self.backpropmask = bsxfun(@eq, output, input_reshape);
                case 'avg' % averaging pooling
                    [output] = mean(input_reshape, 1);
                    self.backpropmask = ones(size(input_reshape)) / self.sizepatch;
                case 'stoic'
                    % a version of stoichastic pooling
                    numavgpool = 1;
                    prob = (self.sizepatch - numavgpool) / self.sizepatch;
                    self.backpropmask = rand(size(input_reshape)) >  self.sizepatch;
                    output = sum(input_reshape .* self.backpropmask, 1);
                    
            end
                    
            output = squeeze(output);
            output = reshape(permute(output, [2,1,3]), self.numchan * self.numpatch, sizebatch);
        end
        
        % some of this can probably be made much faster
        function dLdin = backward(self, dLdout)
            %self.numchan, self.numpatch, sizebatch = size(dLdout)
            [~, ~, sizebatch] = size(dLdout);
            dLdout = reshape(dLdout, self.numpatch, self.numchan, sizebatch);
            % somewhat tricky, the 4 here simply adds a singleton dimension
            dLdout = permute(dLdout, [4,2,1,3]);
            dLdin = bsxfun(@times, dLdout, self.backpropmask);
            % dLdout is now numchan by numpatch by sizebatch in the correct
            % interpretation
        end
    end
    
end