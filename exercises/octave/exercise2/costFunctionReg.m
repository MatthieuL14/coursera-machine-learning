function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

theta2 = theta(2:end);
h = 1 ./ (ones(m, 1)+exp(-X*theta));
J = 1/m * (  -y' * log(h) - (ones(m, 1)-y)' * log(ones(m, 1) - h) ) + (lambda / (2 * m)) * theta2' * theta2;

old_grad = 1/m * X' * (h - y) ;
grad(1) = old_grad(1)
i = 2;
while i <= rows(old_grad),
  grad(i) = old_grad(i) + lambda/m * theta(i)
  i = i+1;
end
grad = grad;

% =============================================================

end
