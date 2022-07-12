function gaussian = createFlatGaussian(X,Y,x, y, sigma, normalize)
gaussian = exp(- ((X-x(1)).^2 + (Y-y(1)).^2)./(2*sigma(1)).^2);
    if normalize
        gaussian = gaussian / sum(gaussian(:));
    end
gaussian = reshape(gaussian, length(X)*length(X), []);
end