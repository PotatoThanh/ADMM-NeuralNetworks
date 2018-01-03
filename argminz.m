function [z] = argminz(a, w, a_in, beta, gamma)
    % This problem is non-convex and non-quadratic (because of the non-linear term h).
    % Fortunately, because the non-linearity h works entry-wise on its argument, the entries
    % in z_l are decoupled. This is particularly easy when h is piecewise linear, as it can
    % be solved in closed form; common piecewise linear choices for h include rectified
    % linear units (ReLUs), that its used here, and non-differentiable sigmoid functions.

    % :param a:    activation matrix (a_l)
    % :param w:    weight matrix (w_l)
    % :param a_in: activation matrix l-1 (a_l-1)
    % :return:     output matrix 

    % m = gather(w* a_in);
    % a = gather(a);
    % beta = gather(beta);
    % gamma = gather(gamma);
    
    m = w* a_in;
    sol1 = (gamma*a + beta*m)/(gamma + beta);
    sol2 = m;

    z1 = gpuArray.zeros(size(a));
    z2 = gpuArray.zeros(size(a));
    z = gpuArray.zeros(size(a));

    z1(sol1>=0) = sol1(sol1>=0);
    % z1 = sol1;
    
    z2(sol2<=0) = sol2(sol2<=0);
    % z2  = sol2;

    fz_1 = gamma*(a - relu(z1)).^2 + beta* ((z1-m).^2); 
    fz_2 = gamma*(a - relu(z2)).^2 + beta* ((z2-m).^2);

    index_z1 = (fz_1<=fz_2);
    index_z2 = (fz_2<fz_1);

    z(index_z1) = z1(index_z1);
    z(index_z2) = z2(index_z2);

    % z = zeros(size(a));

    % for i=1:size(z,1)
    %     for j=1:size(z,2)
    %         z(i,j) = argmin_ez(a(i,j), m(i,j), beta, gamma);
    %     end
    % end
    % z = gpuArray(z);
end