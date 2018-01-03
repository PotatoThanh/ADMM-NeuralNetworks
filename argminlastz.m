function [z] = argminlastz(targets, eps, w, a_in, beta)
    % Minimization of the last output matrix, using the above function.

    % :param targets:  target matrix (equal dimensions of z) (y)
    % :param eps:      lagrange multiplier matrix (equal dimensions of z) (lambda)
    % :param w:        weight matrix (w_l)
    % :param a_in:     activation matrix l-1 (a_l-1)
    % :return:         output matrix last layer

    m = w*a_in;
    z = (targets - eps + beta*m)./(1+beta);
        
end