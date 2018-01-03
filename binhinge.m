function loss = binhinge(z, y)
    %binary hinge error
    
    loss = zeros(size(y));
    
    oneindex = (y==1);
    loss(oneindex) = max(0,1-z(oneindex));
    
    zeroindex = (y==0);
    loss(zeroindex) = max(0,z(zeroindex));
end