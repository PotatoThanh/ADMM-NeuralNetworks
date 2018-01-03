function [lambda] = lambda_update(zl, w, a_in, beta)
% Lagrange multiplier update.

% :param zl:      output matrix last layer (z_L)
% :param w:       weight matrix last layer (w_L)
% :param a_in:    activation matrix l-1 (a_L-1)
% :return:        lagrange update

mpt = w * a_in;

lambda = beta* (zl - mpt);
end