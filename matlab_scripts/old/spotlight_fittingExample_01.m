maindir           = '/Volumes/server/Projects/attentionpRF/Simulations/';

mx = @(x) x / max(x(:)); %Because of the small values obtained when AF and RF are multiplied
% Read in some stimulus apertures
load([fullfile(maindir, 'stimfiles') '/RETBARsmall.mat']);
stim    = logical(stim);
rows    = size(stim,1);
cols    = size(stim,2);
timepoints = size(stim,3);
% Compute the time course
stimVec   = reshape(stim, rows*cols, timepoints);
% Define the visual field positions and time
[X,Y] = meshgrid(linspace(-10,10,cols), linspace(-10,10, rows));
t = (1:timepoints)'; % column vector of time points

% Define a hemodynamic impulse response function
h = t .* exp(-t/5);
h = h / sum(h); % normalize to sum of 1

% Define the stimulus driven RF parameters
nPRFs   = 10;
x       = linspace(-10,10,nPRFs); 
y       = linspace(-10,10,nPRFs); 
sigma   = linspace(1,5,nPRFs); % in degrees of visual field

% Define the attention field parameters
AF_x = 12; AF_y = 0; AF_sigma = 2; %adjusted the AF eccentricity so that it falls outside of the mapped eccentricity

% Create pink noise
whitenoise = randn(timepoints, nPRFs);
pinknoise  = zscore(cumsum(whitenoise));

% Simulate (one tiny) dataset:
[~,AFSimVec] = createGauissanFields(X,Y,AF_x,AF_y,AF_sigma);

noiselessBOLD  = zeros(timepoints, nPRFs);

for ii = 1:nPRFs
    [~,RFSimVec] = createGauissanFields(X,Y,x(ii),y(ii),sigma(ii));
    tmp = conv(AFSimVec .* RFSimVec * stimVec,h);
    tmp = tmp / std(tmp);
    noiselessBOLD(:,ii) = tmp(1:timepoints);
end

data=noiselessBOLD + pinknoise;

%% FITS
%Harvey & Dumoulin -- multiplication of two Gaussians. 
%Attention effects on pRF size: 
%Estimate 3 parameters: 
% 1) stim driven pRF position [x], [y];
% 2) stim driven pRF size [sigma]
% 3) attention field size [AF_sigma]
x0 = [0 , 0, 1]; % Define the initial values
% A = [1,0,0;0,1,0]; % Up to 10deg visual field eccentricity & polar angle:
% b = [10;10];
lb = [-10, -10,  0.01]; % Lower bound
ub = [ 10,  10, 10];    % Lower bound

fitParams= zeros(3,nPRFs);

for ii = 1:nPRFs
    fprintf('Solving pRF for voxel %d of %d\n', ii);
    fun = @(p) (RSSR(p,stim,data(:,ii)));
    fitParams(:,ii) = fmincon(fun,x0,[],[],[],[],lb,ub);
end

% True params
stimParams = [x;y;sigma];
attParamsRight = [AF_x,AF_y,AF_sigma];
attParamsLeft = [-AF_x,AF_y,AF_sigma];

appPRFparamsR = stim2app(stimParams, attParamsRight);
appPRFparamsL = stim2app(stimParams, attParamsLeft);

figure(1);
subplot(131)
scatter(fitParams(1,:),appPRFparamsR(1,:))
subplot(132)
scatter(fitParams(2,:),appPRFparamsR(2,:))
subplot(133)
scatter(fitParams(3,:),appPRFparamsR(3,:))

%% PLOTS
% See how the apparent pRFs change for different stimulus driven pRFs when
% attenional field is kept constant:

apparentpRFs = cell(1,length(x));
stimpRFs = cell(1,length(x));
for ii = 1:length(x)
[apparentpRFs{ii},~] = createGauissanFields(X,Y,fitParams(ii,1),fitParams(ii,2),fitParams(ii,3));
[stimpRFs{ii},~] = createGauissanFields(X,Y,x(ii),y(ii),sigma(ii));
end

for ii = 1:length(x)
    figure (1);
subplot(10,2,2*ii-1)
imagesc(X(:), Y(:), stimpRFs{ii}); axis image xy; 
title('Stim-driven pRF')
subplot(10,2,2*ii)
imagesc(X(:), Y(:), apparentpRFs{ii}); axis image xy; 
title('Apparent pRF (Stim-driven pRF * Attentional field)')
end
set(gcf, 'Position',[0, 1000,1000,1000])
