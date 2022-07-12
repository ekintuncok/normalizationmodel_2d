function [stimdrive, numerator, respSurround, predneuralweights, predsummedweights] = NMA_simulate2D(maindir, params)

stim_ecc    = params(1);
attgain     = params(2);
attx0       = params(3);
atty0       = params(4);
attsd       = params(5);
sigmaNorm   = params(6);
suppweight  = params(7);
summweight  = params(8);
spacegrid   = params(9);


%% Create and scale stimuli
stimtemp = load([fullfile(maindir, 'stimfiles/') 'stim.mat']);
stimorig    = stimtemp.stim(:,:,1:end-1);
stimorig    = logical(stimorig);

[inputStim, X, Y] = createSpatialGrid(stimorig, stim_ecc, spacegrid);
flatten     = @(x) reshape(x, spacegrid*spacegrid, []);
unflatten   = @(x) reshape(x, spacegrid, spacegrid, []);

stim = reshape(inputStim, size(inputStim,1)* size(inputStim,1), []);
%% Simulate RFs
nCenters      = sqrt(size(stim,1));
x             = -1*linspace(-8.8,8.8,nCenters);
y             = linspace(-8.8,8.8,nCenters);
nPRFs         = size(CombVec(y,x),2);
stimdrivenRFs = CombVec(x,y);
baselineresponselevel = 0.05;

% If we keep the x- and y- axes range (look a line up) at [-10 to 10] our
% max eccentricity goes up to 14 degrees. Now we need to assign our sigma
% value accordingly.

for pRFInd = 1:nPRFs
    currCenter = stimdrivenRFs(1:2,pRFInd);
    eccen = sqrt(currCenter(1).^2 + currCenter(2).^2);
    stimdrivenRFs(3,pRFInd) = 0.05 + 0.1*eccen;
end

%% Stimulus and Suppressive Drive
stimdrive    = zeros(size(stimdrivenRFs,2),size(inputStim,3));
numerator = zeros(size(stimdrivenRFs,2),size(inputStim,3));
respSurround = zeros(size(stimdrivenRFs,2),size(inputStim,3));

for ii = 1:size(inputStim,3)
    for rfind = 1:size(stimdrivenRFs,2)
        RF = createFlatGaussian(X,Y,stimdrivenRFs(2,rfind), stimdrivenRFs(1,rfind),...
            stimdrivenRFs(3,rfind), 1);
        stimdrive(rfind, ii) = RF'*stim(:,ii)+baselineresponselevel;
        afweight = exp(-((attx0-stimdrivenRFs(2,rfind)).^2+(atty0-stimdrivenRFs(1,rfind)).^2)./(2*attsd).^2);
        afweight = attgain*afweight+1;
        numerator(rfind,ii) = stimdrive(rfind,ii).*afweight;
    end
    for rfsuppind = 1:size(stimdrivenRFs,2)
        distance = sqrt((X-stimdrivenRFs(2,rfsuppind)).^2+(Y-stimdrivenRFs(1,rfsuppind)).^2);
        supp = exp(-.5*(distance/(stimdrivenRFs(3,rfsuppind)*suppweight)).^2);
        supp = supp / sum(supp(:));
        flatsurr = reshape(supp, length(supp)*length(supp), []);
        respSurround(rfsuppind,ii) = flatsurr' * numerator(:,ii);
    end
end

%% population response
predneuralweights = numerator ./ (respSurround + sigmaNorm);
predsummedweights = zeros(size(stimdrivenRFs,2),size(inputStim,3));

for ii = 1:size(predneuralweights,2)
    for summidx = 1:size(stimdrivenRFs,2)
        distance = sqrt((X-stimdrivenRFs(2,summidx)).^2+(Y-stimdrivenRFs(1,summidx)).^2);
        summation = exp(-.5*(distance/(stimdrivenRFs(3,summidx)*summweight)).^2);
        summation = reshape(summation, length(summation)*length(summation), []);
        predsummedweights(summidx,ii) = summation' * predneuralweights(:,ii); 
    end
end
end
