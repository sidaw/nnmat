actfuncs = containers.Map;
actgrads = containers.Map;

actfuncs('sigmoid') = @(x) 1./(1+exp(-x));
actgrads('sigmoid') = @(x,y) y.*(1-y);

actfuncs('identity') = @(x) x;
actgrads('identity') = @(x,y) ones(size(x));

actfuncs('relu') = @(x) max(0,x);
actgrads('relu') = @(x,y) (x>0)*1;