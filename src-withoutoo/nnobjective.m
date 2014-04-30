[obj grad] = function nn2obj(params, X, y)

dimdata =  2;
numhid = 3;
numclass = 1;
numdata = 1;
L = {}

L{1} = DenseLinearLayer(dimdata, numhid)
L{2} = SigmoidLayer(numhid)
L{3} = DenseLinearLayer(numhid, numclass)

data = randn(dimdata, numdata);
y = 10*randn(numclass, numdata)

currentIn = data;
for i = 1:length(L)
  L{i}.input = currentIn;
  L{i}.output = L{i}.forward(L{i}, currentIn);
  currentIn = L{i}.output;
end

obj = sum( (L{3}.output - y).^2 )

dLdout = L{3}.output - y;

dparams = {};
for i = length(L):-1:1
  [dLdout, dparam] = L{i}.backward(L{i}, dLdout);
  dparams{i} = dparam;
  L{i}.params = L{i}.params - 1*dparams{i};
end

grad = params2stack()

