function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

  % makes a 47x1 matrix in example case
  h = X * theta;

  % need to transpose right side of subtraction to make 3x1 matrice to match theta
  theta = theta - (alpha/m .* ((h - y)' *X))';
  
  % Save the cost J in every iteration    
  J_history(iter) = computeCostMulti(X, y, theta);

end

end
