function [w] = weight_update(layer_output, activation_input, rho)
    % Consider it now the minimization of the problem with respect to W_l.
    % For each layer l, the optimal solution minimizes ||z_l - W_l a_l-1||^2. This is simply
    % a least square problem, and the solution is given by W_l = z_l p_l-1, where p_l-1
    % represents the pseudo-inverse of the rectangular activation matrix a_l-1.

    % :param layer_output:        output matrix (z_l)
    % :param activation_input:    activation matrix l-1  (a_l-1)
    % :return:                    weight matrix
    w = layer_output * pinv(activation_input - rho);
end