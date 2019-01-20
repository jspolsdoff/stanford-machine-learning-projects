function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% take the matrix of theta and multiple by the features
predictions = X*theta;

% subtract the actual value from the prediction and square it
sqrErrors = (predictions - y).^2;

J = 1/(2*m) * sum(sqrErrors);

end
