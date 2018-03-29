function result = argmin_ez( a, m, beta, gamma)
    %sol = (gamma*a + beta*m)/(gamma + beta);
    sol1 = (gamma*a + beta*m)/(gamma + beta);
    sol2 = m;
    
    z1 = 0;
    z2 = 0;

    if sol1>=0
        z1 = sol1;
    else
        z1 = 0;
    end

    if sol2<=0
        z2 = sol2;
    else
        z2 = 0;
    end

    fz_1 = gamma*(a - relu(z1))^2 + beta* ((z1-m)^2); 
    fz_2 = gamma*(a - relu(z2))^2 + beta* ((z2-m)^2);
    if fz_1<=fz_2
        result = z1;
    else
        result = z2;
    end
%     if m>=0
%         if sol>0
%            result = sol;
%            return;
%         else
%             result = 0;
%             return;
%         end
%     else        
%         if a<=0
%             result = m;
%             return;
%         end
%         res1 = (gamma * ((a - (max(0, sol)))^2)) + (beta * ((sol - m)^2));
%         res2 = gamma * a^2;
%         if res1<=res2
%             result = sol;
%             return;
%         end
%         result = m;
%         return;
%     end
end

