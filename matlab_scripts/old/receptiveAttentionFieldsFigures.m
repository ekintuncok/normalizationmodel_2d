clear variables
clc
addpath('/Users/ekintuncok/Desktop/pRF_AF Model Fit');
mx = @(x) x / max(x(:)); %Because of the small values obtained when AF and RF are multiplied

% Read in some stimulus apertures
load('RETBARsmall.mat', 'stim');
stim        = logical(stim);
rows        = size(stim,1);
cols        = size(stim,2);
timepoints  = size(stim,3);
% Compute the time course
stimVec     = reshape(stim, rows*cols, timepoints);

% Define the visual field positions and time
[X,Y] = meshgrid(linspace(-10,10,cols), linspace(-10,10, rows));
t = 1:timepoints;
% Define a hemodynamic impulse response function
h = t .* exp(-t/5);
h = h / sum(h); % normalize to sum of 1
 
% Define the stimulus driven RF
x = -2; y = 3; sigma = 2; % in degrees of visual field
[RF,RFvec] = createGauissanFields(X,Y,x,y,sigma);

% Define the attention field for the left and the right visual field
AF_x = [-9, 9]; AF_y = 0; AF_sigma = 1;
AF.left = createGauissanFields(X,Y,AF_x(1),AF_y,AF_sigma);
AF.right = createGauissanFields(X,Y,AF_x(2),AF_y,AF_sigma);

AF_leftvec = reshape(AF.left,  1, rows*cols);
AF_rightvec = reshape(AF.right,  1, rows*cols);

% Simulate a dataset with a slightly different AF position and size to
% calculate the fits:y
[~,AFSimVec] = createGauissanFields(X,Y,AF_x(1)+4*rand(1),AF_y+2*rand(1),AF_sigma+2*rand(1));
[~,RFSimVec] = createGauissanFields(X,Y,x(1)+4*rand(1),y+2*rand(1),sigma+2*rand(1));
boldMeasured4  = conv(AFSimVec .* RFSimVec * stimVec,h);
boldMeasured4  = mx(boldMeasured4(1:timepoints))+ pinknoise(length(t))';

% Predict the neural response assuming no attention
predNeural.attend0  = RFvec * stimVec;
predBOLD.attend0    = conv(predNeural.attend0, h);
predBOLD.attend0    = mx(predBOLD.attend0(1:timepoints))+ pinknoise(length(t))';

% Predict the neural response assuming Attend LEFT
predNeural.attendL  = AF_leftvec .* RFvec * stimVec;
predBOLD.attendL    = conv(predNeural.attendL, h);
predBOLD.attendL    = mx(predBOLD.attendL(1:timepoints))+ pinknoise(length(t))';

% Predict the neural response assuming Attend RIGHT
predNeural.attendR  = AF_rightvec .* RFvec * stimVec;
predBOLD.attendR    = conv(predNeural.attendR, h);
predBOLD.attendR    = mx(predBOLD.attendR(1:timepoints))+ pinknoise(length(t))';

%% PLOTS
% Plot the time course
figure(1)
subplot(1,2,1)
plot(t, mx(predNeural.attend0), t, mx(predNeural.attendL), t, mx(predNeural.attendR),'LineWidth', 1);  
legend('Attend 0', 'Attend Left', 'Attend Right')
title('Predicted neural response')

subplot(1,2,2)
plot(t, predBOLD.attend0, t, predBOLD.attendL, t, predBOLD.attendR,'LineWidth', 1);  
legend('Attend 0', 'Attend Left', 'Attend Right')
title('Predicted BOLD response')

% Plot the RFs 
figure(2); clf

subplot(3,2,1)
imagesc(X(:), Y(:), RF); axis image xy; 
title('STIMULUS RF')

subplot(3,2,3)
imagesc(X(:), Y(:), AF.right); axis image xy; 
title('RIGHT attentional field')

subplot(3,2,5)
imagesc(X(:), Y(:), AF.left); axis image xy; 
title('LEFT attentional field')

subplot(3,2,2)
imagesc(X(:), Y(:), RF); axis image xy; 
title('STIMULUS RF')

subplot(3,2,4)
imagesc(X(:), Y(:), AF.right.*RF); axis image xy; 
title('RF, attend RIGHT')

subplot(3,2,6)
imagesc(X(:), Y(:), AF.left.*RF); axis image xy; 
title('RF, attend LEFT')

% Make a movie 
figure(2); clf
% Plot the stimulus and time course
for j = 1:length(x)
for ii = 1:timepoints
    subplot(2,2,2*j-1)
    imagesc(X(:), Y(:), stim(:,:,ii)+RF{j} ./ max(RF{j}(:)), [0 1]), axis square xy
    title(ii); pause(0.01); 
    
    subplot(2,2,2*j)
    plot(1:ii, predNeural{j}(1:ii), '-', 1:ii, predBOLD{j}(1:ii), '-','LineWidth', 3); 
    xlim([0 timepoints]); ylim([0 1]); axis square; 
end
end


function [RF, RFvec] = createGauissanFields(X,Y,x, y, sigma)
RF = exp(-((X-x(1)).^2 + (Y-y(1)).^2)./(2*sigma(1)).^2);
RF = RF / sum (RF(:)); % normalize to sum of 1
RFvec       = reshape(RF,  1, length(X)*length(Y));
end

function [resSumSq] = RSSR(x,sigma, AF_x, AF_sigma, xMeasured)
resSumSq = ((x*AF_sigma.^2) + AF_x*sigma.^2/(AF_sigma.^2+sigma.^2) - xMeasured).^2;
end
