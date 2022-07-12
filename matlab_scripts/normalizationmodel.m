function [normalizedresponse, predsummedweights] = normalizationmodel(stimorig, stim_ecc, stimdrivenRFs, attx0, atty0, attsd, attgain, sigmaNorm)
% % pad stimulus with zeros to avoid edge artifacts
gridsize = sqrt(length(stimdrivenRFs));
baselineresponselevel = 0.05;
[inputStim, X, Y] = createSpatialGrid(stimorig, stim_ecc, gridsize);
stim = reshape(inputStim, size(inputStim,1)* size(inputStim,1), []);

% preallocate
stimdrive = zeros(size(stimdrivenRFs,2),size(inputStim,3));
numerator = zeros(size(stimdrivenRFs,2),size(inputStim,3));
respSurround = zeros(size(stimdrivenRFs,2),size(inputStim,3));
predsummedweights = zeros(size(stimdrivenRFs,2),size(inputStim,3));

%% Attention field:
attfield = createFlatGaussian(X,Y,attx0, atty0, attsd, 0);
attfield = attgain*attfield  + 1;

%% Stimulus and Suppressive Drive
for ii = 1:size(inputStim,3)
    for rfind = 1:size(stimdrivenRFs,2)
        % get the stim driven RF
        RF = createFlatGaussian(X,Y,stimdrivenRFs(2,rfind), stimdrivenRFs(1,rfind),...
            stimdrivenRFs(3,rfind), 1);
        stimdrive(rfind, ii) = RF'*stim(:,ii)+baselineresponselevel;
    end
    numerator(:,ii) = stimdrive(:,ii).*attfield;
    for rfsuppind = 1:size(stimdrivenRFs,2)
        distance = sqrt((X-stimdrivenRFs(2,rfsuppind)).^2+(Y-stimdrivenRFs(1,rfsuppind)).^2);
        supp = exp(-.5*(distance/(stimdrivenRFs(3,rfsuppind)*3)).^2);
        supp = supp / sum(supp(:));
        flatsurr = reshape(supp, length(supp)*length(supp), []);
        respSurround(rfind,ii) = flatsurr' * numerator(:,ii);
    end
end

%% population response
normalizedresponse = numerator ./ (respSurround + sigmaNorm);
for ii = 1:size(predneuralweights,2)
    for summidx = 1:size(stimdrivenRFs,2)
        spatialsumm = createFlatGaussian(X,Y,stimdrivenRFs(2,summidx), stimdrivenRFs(1,summidx),...
            stimdrivenRFs(3,summidx), 1);
        predsummedweights(summidx,ii) = spatialsumm' * normalizedresponse(:,ii);
    end
end
end