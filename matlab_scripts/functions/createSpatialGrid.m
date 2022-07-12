function [inputStim, X, Y] = createSpatialGrid(stim, stim_ecc, gridsize)

ecc = linspace(-stim_ecc,stim_ecc,gridsize);
[X,Y] = meshgrid(ecc);
Y = -1*Y;
inputStim = zeros(size(X,1),size(X,1),size(stim,3));
fullSize = size(X,1);
stimSize = fullSize-(0.25*fullSize);
imStart = fullSize/2-(stimSize/2);
imEnd = imStart+stimSize-1;
imIdx = imStart:imEnd;
for s = 1:size(stim,3)
    inputStim(imIdx,imIdx,s) = imresize(stim(:,:,s),[stimSize stimSize],'nearest');
end

end