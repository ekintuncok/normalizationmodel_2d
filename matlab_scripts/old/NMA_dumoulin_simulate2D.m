function predneuralweights = NMA_dumoulin_simulate2D(maindir, params)

mxecc       = params(1);
RFsd        = params(2);
attgain     = params(3);
attx0       = params(4);
atty0       = params(5);
attsd       = params(6);
pooledPopRespsd = params(7);
sigmaNorm       = params(8);
normRFScalar = 2; % normalization RF scalar: if 2, then normRF is twice as wide as stimdriveRF

%% Space (1D)
ecc = linspace(-5,5,64);
[X,Y] = meshgrid(ecc);

%% Stimuli
load([fullfile(maindir, 'stimfiles/') 'stim.mat'])
stim    = stim(:,:,1:end-1);
stim    = logical(stim);
% pad stimulus with zeros to avoid edge artifacts
inputStim = zeros(size(X,1),size(X,1),size(stim,3));
fullSize = size(X,1);
stimSize = fullSize-(0.25*fullSize);
imStart = fullSize/2-(stimSize/2);
imEnd = imStart+stimSize-1;
imIdx = imStart:imEnd;
for s = 1:size(stim,3)
    inputStim(imIdx,imIdx,s) = imresize(stim(:,:,s),[stimSize stimSize],'nearest');
end

%% Neural RFs
% suppressive pool
RFsupp = exp(- ((X-0).^2 + (Y-0).^2)./(2*1.5*RFsd).^2); RFsupp = RFsupp./sum(RFsupp(:)); % normalize to unit volume (this is critical)
nCenters    = size(inputStim,1);
x           = linspace(-8.8,8.8,nCenters); 
y           = linspace(-8.8,8.8,nCenters);
nPRFs       = size(CombVec(x,y),2);
numSigmas   = 10;
sigma       = linspace(0.1,5,numSigmas); % in degrees of visual field
maxEccen    = sqrt(x(1).^2 + y(1).^2);
sampleEccen = linspace(0,maxEccen,numSigmas);

% If we keep the x- and y- axes range (look a line up) at [-10 to 10] our
% max eccentricity goes up to 14 degrees. Now we need to assign our sigma
% value accordingly. 
stimdrivenRFs   = zeros(3,size(CombVec(x,y),2));
stimdrivenRFs(1:2,:) = CombVec(x,y);

for pRFInd = 1:nPRFs
    currCenter = stimdrivenRFs(1:2,pRFInd);
    % calculate the euclidean distance from the center of gaze (origin) to
    % assign the sigma value based on the distance. The distance here is a
    % proxy of eccentricity (it's actually the literal definition of it!)
    eccen = sqrt(currCenter(1).^2 + currCenter(2).^2);

    % create smoothly varying sigmas across eccentricity for each pRF
    stimdrivenRFs(3,pRFInd) = 0.05 + 0.1*eccen;
end

%% Attention field
attfield = exp(-((X-attx0).^2 +(Y-atty0).^2)./(2*attsd).^2);
attfield = attgain*attfield  + 1;

%% Stimulus and Suppressive Drive
stimdrive = zeros(size(stimdrivenRFs,2),size(inputStim,3));
numerator = zeros(size(stimdrivenRFs,2),size(inputStim,3));

for rfind = 1:size(stimdrivenRFs,2)
    for ii = 1:size(inputStim,3)
        % get the stim driven RF 
        RF = exp(-((X-(stimdrivenRFs(1,rfind))).^2 + ...
            (Y-(stimdrivenRFs(2,rfind))).^2)./(2*(stimdrivenRFs(3,rfind))).^2);
         RF = RF./sum(RF(:)); % unit volume

        % get the normalization RF
        RFnorm = exp(-((X-(stimdrivenRFs(1,rfind))).^2 + ...
            (Y-(stimdrivenRFs(2,rfind))).^2)./(normRFScalar*2*(stimdrivenRFs(3,rfind))).^2);
         RFnorm = RFnorm./sum(RFnorm(:)); % unit volume

        % impose the attention field
        RF = RF .* attfield;
        %RFnorm = RFnorm .* attfield;

        % get the stimulus vectorized
        stim = inputStim(:,:,ii);
        stim = stim(:);
        numerator(rfind, ii)     = RF(:)'*stim;
        numeratorIm              = RF.*reshape(stim,size(X));
        denominator(rfind, ii)   = sum(RFnorm(:).*numeratorIm(:));
    end
end
numerator_pop = zeros(size(inputStim,1),size(inputStim,2),size(inputStim,3));
denominator_pop = zeros(size(inputStim,1),size(inputStim,2),size(inputStim,3));

for s = 1:size(stimdrive,2)
    numerator_pop(:,:,s) = reshape(numerator(:,s), [size(inputStim,1) size(inputStim,2)]);
    denominator_pop(:,:,s) = reshape(denominator(:,s), [size(inputStim,1) size(inputStim,2)]);
end

%% population response
sptPopResp = numerator_pop ./ (denominator_pop + sigmaNorm);
%close all, for ii = 1:size(sptPopResp,3), imagesc(sptPopResp(:,:,ii)), colormap gray; pause(0.5), end

% Go across time for a specific location in the pop response,let's say x =
% 1, y = 2. 
predneuralweights = reshape(sptPopResp,[size(sptPopResp,1)*size(sptPopResp,2) size(sptPopResp,3)]);
end

