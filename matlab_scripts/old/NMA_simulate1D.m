function [x, popresp, summation] = NMA_simulate1D(params)

npoints     = params(2);
mxecc       = params(3);
RFsd        = params(4);
attgain     = params(5);
attx0       = params(6);
attsd       = params(7);
summationsd = params(8);
sigma       = params(9);
visualize   = params(10);

%% Space (1D)
x = linspace(-mxecc,mxecc,npoints)'; % degrees
%% Neural RFs
RF = exp(-1/2*(x/RFsd).^2);
RF = RF / sum(RF);
RFsupp = exp(-1/2*(x/(RFsd*3)).^2);
RFsupp = RFsupp / sum(RFsupp);
%% Voxel summation (across neurons)
RFsumm = exp(-1/2*(x/summationsd).^2);
RFsumm = RFsumm / sum(RFsumm);
%% Stimuli
stim = zeros(npoints,npoints);
for ind = 1:length(stim)
    stim(ind:ind+5,ind:ind+5) = 1;
end
stim = stim(1:npoints,1:npoints);
%% Attention field
attfield = exp(-1/2*((x-attx0)/attsd).^2);
attfield = attgain*attfield  + 1;
%% Stimulus and Suppressive Drive
stimdrive = conv2(stim, RF, 'same');
numerator = stimdrive .* (attfield * ones(1,npoints));
suppdrive = conv2(numerator, RFsupp, 'same');

%% population response
popresp = numerator ./ (suppdrive + sigma);
summation = conv2(popresp, RFsumm, 'same');
if visualize
    fH = figure; clf;
    lw = 3;
    fs = 10;
    subplot(2,3,1); plot(x, RF, x, RFsupp, 'LineWidth', lw);
    title('RF, center and surround');
    plotOptions(gca, fs);
    subplot(2,3,2); plot(x, attfield, 'LineWidth', lw);
    title('Attention field');
    plotOptions(gca, fs);
    subplot(2,3,3);
    imagesc(x, x, stim')
    title('Stimuli');
    plotOptions(gca, fs);
    subplot(2,3,4);
    imagesc(x, x, stimdrive);
    title('Stimulus drive');
    plotOptions(gca, fs);
    ylabel('Neural population')
    xlabel('Stimulus')
    subplot(2,3,5);
    imagesc(x, x, suppdrive);
    title('Suppressive drive');
    plotOptions(gca, fs);
    ylabel('Neural population')
    xlabel('Stimulus')
    subplot(2,3,6);
    imagesc(x, x, popresp);
    title('Population response');
    plotOptions(gca, fs);
    ylabel('Neural population')
    xlabel('Stimulus')
    set(gcf,'Position',[0 0 600 350])
end
end