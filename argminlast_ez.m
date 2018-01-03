function [result] = argminlast_ez(y, eps, m, beta)
    result = (y - eps + beta*m)/(1+beta);
    
    % w = m - (eps / (2 * beta));
    % s0 = m - ((1 + eps) / (2 * beta));
    % s1 = m + ((1 - eps) / (2 * beta));

    % if y==0
    %     if s0 + w >=0
    %         result = s0;
    %         return;
    %     end
    %     result = w;
    %     return;
    % else
    %     if s1 + w <=2
    %         result = s1;
    %         return;
    %     end
    %     result = w;
    %     return;
    % end
end