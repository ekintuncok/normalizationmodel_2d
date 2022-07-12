function [appPRFparams] = stim2app(stimParams, attParams)
% [appPRFparams] = stim2app(stimParams, attParams)
% Derive the x,y,sigma parameters of the apparent pRF from the parameters
% of the stimulus pRF and attentional field
%
% Inputs:
%   stimparams: 3 x n matrix of pRF parameters (x,y,sigma) by voxel
%   attParams: 3 element array of attentional field parameters (x,y,sigma)
% Output
%   appPRFparams: 3 x n matrix of apparent pRF parameters (x,y,sigma)
%
% Example:
% appPRFparams = stim2app([2 0 1], [-2 0 2]);
%
% Formula from Klein et al, 2014 (DOI: )

appPRFparams = zeros(size(stimParams));

x1 = stimParams(1,:);
y1 = stimParams(2,:);
s1 = stimParams(3,:);

x2 = attParams(1);
y2 = attParams(2);
s2 = attParams(3);

appPRFparams(1,:) = ((x1*s2^2)+(x2*s1.^2))./(s1.^2+s2^2);
appPRFparams(2,:) = ((y1*s2^2)+(y2*s1.^2))./(s1.^2+s2^2);
appPRFparams(3,:) = (s1.^2*s2^2)./(s1.^2+s2^2);
end