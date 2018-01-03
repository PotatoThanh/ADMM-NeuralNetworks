function [a] = activation_update(next_weight, next_layer_output, layer_nl_output, beta, gamma)
% Minimization for a_l is a simple least squares problem similar to the weight update.
% However, in this case the matrix appears in two penalty terms in the problem, and so
% we must minimize:

%     beta ||z_l+1 - W_l+1 a_l||^2 + gamma ||a_l - h(z_l)||^2

% :param next_weight:         weight matrix l+1 (w_l+1) 
% :param next_layer_output:   output matrix l+1 (z_l+1)
% :param layer_nl_output:     activate output matrix h(z) (h(z_l)) 
% :return:                    activation matrix

% calculate relu
layer_nl_output = relu(layer_nl_output);

% activation inverse
m1 = beta*(next_weight' * next_weight);
m2 = gamma * eye(size(m1));
av = inv(m1 + m2);

% activation formulate
m3 = beta * (next_weight' * next_layer_output);
m4 = gamma * layer_nl_output;
af = m3 + m4;

%output
a = av*af;
    
end