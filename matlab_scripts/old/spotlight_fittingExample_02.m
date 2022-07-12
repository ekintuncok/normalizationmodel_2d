%%%%%% ATTENTION PRF - SIMULATIONS %%%%%
%%% ET & JW
%%% last update: 02/2022
setNum = 2;
saveDir = sprintf('/Volumes/server/Projects/attentionpRF/Simulations/Attention_pRF/set%i_output/', setNum);
maindir           = '/Volumes/server/Projects/attentionpRF/Simulations/';

if ~exist(saveDir)
    mkdir(saveDir);
end
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
stim_ecc = 10;
step_size = stim_ecc/5;

rF_size_major = step_size:step_size:stim_ecc;
x = -stim_ecc:step_size:stim_ecc;
y = -stim_ecc:step_size:stim_ecc;

RFIndices = combvec(rF_size_major,x,y);
nPRFs     = length(RFIndices);
% Define the attention field parameters
% AF parameters here are set for Gabor targets (AFs) that are positioned
% along the horizontal meridian. Therefore, Y is always equal to 0.
AF_x    = -12:4:12;
AF_y    = 0;
AF_sigma = 1:2:8;
AFIndices = combvec(AF_sigma,AF_x,AF_y);
nAFs   = length(AFIndices);

% Simulate (one tiny) dataset with pink noise added:

for af = 1:length(AFIndices)
    [~,AFSimVec] = createGauissanFields(X,Y,AFIndices(2,af),AFIndices(3,af),AFIndices(1,af));
    for rf = 1:length(RFIndices)
        [~,RFSimVec] = createGauissanFields(X,Y,RFIndices(2,rf),RFIndices(3,rf),RFIndices(1,rf));
        tmp = conv(AFSimVec .* RFSimVec * stimVec,h);
        tmp = tmp / std(tmp);
        noiselessBOLD{af}(:,rf) = tmp(1:timepoints);
        whitenoise = randn(timepoints, 1);
        pinknoise  = zscore(cumsum(whitenoise));
        data{af}(:,rf) = noiselessBOLD{af}(:,rf) + pinknoise;
    end
end
save([saveDir 'simulation_data_set1.mat'], 'data');
save([saveDir 'RFIndices_set1.mat'], 'RFIndices');
save([saveDir 'AFIndices_set1.mat'], 'AFIndices');
%
%
%% FITS
%Harvey & Dumoulin -- multiplication of two Gaussians.
%Attention effects on pRF size:
%Estimate 3 parameters:
% 1) stim driven pRF position [x], [y];
% 2) stim driven pRF size [sigma]
% 3) attention field size [AF_sigma]
x0 = [0 , 0, 4]; % Define the initial values
% A = [1,0,0;0,1,0];
% b = [10;10];
lb = [-12, -12, 0.01]; % Lower bound
ub = [ 12,  12, 8];    % Lower bound

fitParams= cell(nAFs,1);

for af = 1:nAFs
    for rf = 1:nPRFs
        fprintf('Solving pRF for voxel %d of attention field %d\n', rf, af);
        fun = @(p) (RSSR(p,stim,data{af}(:,rf)));
        fitParams{af}(:,rf) = fmincon(fun,x0,[],[],[],[],lb,ub);
    end
end
save([saveDir 'RFIndices_set1.mat'], 'fitParams')


% % True params
% stimParams = [x;y;sigma];
% attParamsRight = [AF_x; repmat(AF_y,1,20); repmat(AF_sigma,1,20)];
% attParamsLeft = [-AF_x; repmat(AF_y,1,20); repmat(AF_sigma,1,20)];
% 
% for ii = 1:nAFs
% appPRFparamsR{ii} = stim2app(stimParams, attParamsRight(:,ii));
% appPRFparamsL{ii} = stim2app(stimParams, attParamsLeft(:,ii));
% end
%  %% Plot the pRF fits and model prediction
%  % Find a better way to index each subplot (npRFs might and will change)
% for ii = 1:length(attParamsRight)
%     if (1 <= ii) &&  (ii <= 5)
% figure(1);
% subplot(5, 3, 2*ii+ii-2)
% scatter(fitParams{ii}(1,:),appPRFparamsR{ii}(1,:))
% xlabel('pRF fit X Coord')
% ylabel('AF Model prediction X Coord')
% title(num2str(attParamsRight(1,ii)))
% subplot(5, 3, 2*ii+ii-1)
% scatter(fitParams{ii}(2,:),appPRFparamsR{ii}(2,:))
% xlabel('pRF fit Y Coord')
% ylabel('AF Model prediction Y Coord')
% title(num2str(attParamsRight(2,ii)))
% subplot(5, 3, 3*ii)
% scatter(fitParams{ii}(3,:),appPRFparamsR{ii}(3,:))
% xlabel('pRF fit Sigma')
% ylabel('AF Model prediction Sigma')
% title(num2str(attParamsRight(3,ii)))
% hold on
%     end
%     if (6 <= ii) &&  (ii <= 10)
%         jj = ii-5;
% figure(2);
% subplot(5, 3, 2*jj+jj-2)
% scatter(fitParams{ii}(1,:),appPRFparamsR{ii}(1,:))
% xlabel('pRF fit X Coord')
% ylabel('AF Model prediction X Coord')
% title(num2str(attParamsRight(1,ii)))
% subplot(5, 3, 2*jj+jj-1)
% scatter(fitParams{ii}(2,:),appPRFparamsR{ii}(2,:))
% xlabel('pRF fit Y Coord')
% ylabel('AF Model prediction Y Coord')
% title(num2str(attParamsRight(2,ii)))
% subplot(5, 3, 3*jj)
% scatter(fitParams{ii}(3,:),appPRFparamsR{ii}(3,:))
% xlabel('pRF fit Sigma')
% ylabel('AF Model prediction Sigma')
% title(num2str(attParamsRight(3,ii)))
% hold on
%     end
%     if (11 <= ii) && (ii <= 15)
%         jj = ii-10;
% figure(3);
% subplot(5, 3, 2*jj+jj-2)
% scatter(fitParams{ii}(1,:),appPRFparamsR{ii}(1,:))
% xlabel('pRF fit X Coord')
% ylabel('AF Model prediction X Coord')
% title(num2str(attParamsRight(1,ii)))
% subplot(5, 3, 2*jj+jj-1)
% scatter(fitParams{ii}(2,:),appPRFparamsR{ii}(2,:))
% xlabel('pRF fit Y Coord')
% ylabel('AF Model prediction Y Coord')
% title(num2str(attParamsRight(2,ii)))
% subplot(5, 3, 3*jj)
% scatter(fitParams{ii}(3,:),appPRFparamsR{ii}(3,:))
% xlabel('pRF fit Sigma')
% ylabel('AF Model prediction Sigma')
% title(num2str(attParamsRight(3,ii)))
%     end
%     if (16 <= ii) &&  (ii <= 20)
%         jj = ii-15;
% figure(4);
% subplot(5, 3, 2*jj+jj-2)
% scatter(fitParams{ii}(1,:),appPRFparamsR{ii}(1,:))
% xlabel('pRF fit X Coord')
% ylabel('AF Model prediction X Coord')
% title(num2str(attParamsRight(1,ii)))
% subplot(5, 3, 2*jj+jj-1)
% scatter(fitParams{ii}(2,:),appPRFparamsR{ii}(2,:))
% xlabel('pRF fit Y Coord')
% ylabel('AF Model prediction Y Coord')
% title(num2str(attParamsRight(2,ii)))
% subplot(5, 3, 3*jj)
% scatter(fitParams{ii}(3,:),appPRFparamsR{ii}(3,:))
% xlabel('pRF fit Sigma')
% ylabel('AF Model prediction Sigma')
% title(num2str(attParamsRight(3,ii)))
%     end
% end
% %% PLOTS
% % % See how the apparent pRFs change for different stimulus driven pRFs when
% % % attenional field is kept constant:
% %
% % apparentpRFs = cell(1,length(x));
% % stimpRFs = cell(1,length(x));
% % for ii = 1:length(x)
% % [apparentpRFs{ii},~] = createGauissanFields(X,Y,fitParams(1,ii),fitParams(2,ii),fitParams(3,ii));
% % [stimpRFs{ii},~] = createGauissanFields(X,Y,appPRFparamsR(ii),appPRFparamsR(ii),AF_sigma);
% % end
% %
% % for ii = 1:20
% %     figure (1);
% % subplot(10,2,2*ii-1)
% % imagesc(X(:), Y(:), stimpRFs{ii}); axis image xy;
% % title('Stim-driven pRF')
% % subplot(10,2,2*ii)
% % imagesc(X(:), Y(:), apparentpRFs{ii}); axis image xy;
% % title('Apparent pRF (Stim-driven pRF * Attentional field)')
% % end
% % set(gcf, 'Position',[0, 1000,1000,1000])
