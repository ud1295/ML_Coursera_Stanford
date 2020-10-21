function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)


X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================

temp1 = (X * Theta' - Y);
temp = temp1.^2;
J = sum(sum(temp(R == 1)))/ 2;
J = J + lambda/ 2 * sum(sum(Theta.^2)) + lambda / 2 * sum(sum(X.^2));
for i = 1 : num_movies
    idx = find(R(i, :) == 1);
    Theta_temp = Theta(idx, :);  
    Y_tem = Y(i, idx); 
    X_grad(i, :) = (X(i, :) * Theta_temp' - Y_tem) * Theta_temp + lambda * X(i,:);
    
end

for j = 1: num_users
    idy = find(R(:, j) == 1);  
    Theta_temp = Theta(j, :); 
    Y_tem = Y(idy, j); 
    X_tem = X(idy, :);
    Theta_grad(j,:) = (X_tem * Theta_temp' - Y_tem)' * X_tem + lambda * Theta(j, :);
end

grad = [X_grad(:); Theta_grad(:)];
end