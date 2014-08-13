function diff = derivativeCheck(funObj,x,order,type,varargin)
% diff = derivativeCheck(funObj,x,order,useComplex,varargin)
%
% type = 1 (simple forward-difference)
% type = 2 (central differencing - default)
% type = 3 (complex-step deriative)

if nargin < 3
	order = 1; % Only check gradient by default
	if nargin < 4
		type = 2; % Use central-differencing by default
	end
end

if order == 2
	[f,g,H] = funObj(x,varargin{:});
	
	fprintf('Checking Hessian...\n');
	[f2,g2,H2] = autoHess(x,type,funObj,varargin{:});
	
	fprintf('Max difference between user and numerical hessian: %e\n',max(abs(H(:)-H2(:))));
	if max(abs(H(:)-H2(:))) > 1e-4
		H
		H2
		diff = abs(H-H2)
		pause;
	end
else
	[f,g] = funObj(x,varargin{:});
    p = length(g);
    fprintf('Just checking 100 fixed gradients out of %d\n', p)
    interval = floor(p/100);
    l = length(p);
    
    fprintf(' checking 100 fixed gradients out of %d\n', p)
    interval = floor(p/100);
    randomlist = randi(p,[1,100]);
    fprintf('checking 100 random gradients out of %d\n', p)
    listtocheck = [randomlist [1:50,p-50:p]];

	g = g(listtocheck);
	fprintf('Checking Gradient...\n');
	[f2,g2] = autoGrad(x,listtocheck, type,funObj,varargin{:});
    
	
	fprintf('Max difference between user and numerical gradient: %e\n',max(abs(g-g2)));
	if max(abs(g-g2)) > 1e-4
		fprintf('User NumDif:\n');
		[g g2]
		diff = abs(g-g2)
		pause
	end
end

