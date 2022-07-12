function [RF, RFvec] = createRFs(X,Y,x, y, sigma)
RF = exp(- ((X-x(1)).^2 + (Y-y(1)).^2)./(2*sigma(1)).^2);
RF = RF / sum (RF(:)); % normalize to sum of 1
RFvec = reshape(RF,  1, length(X)*length(Y));
end