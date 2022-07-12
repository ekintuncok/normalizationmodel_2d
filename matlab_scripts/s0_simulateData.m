maindir = './';
addpath(genpath(maindir))

mxecc       = 10;
attgain     = 4;
attx0locs   = [0, 0, -5, 5];
atty0locs   = [5, -5, 0, 0];
attsd       = 1.5;
sigma       = 0.1;
suppweight  = 3;
summweight  = 3;
spacegrid   = 64;

params      = [mxecc,1,0,0,3,sigma,suppweight,summweight,spacegrid];
[basestimdrive, basenumerator, basesupp, baselinepredneuralweights, baselinepredsummedweights] = NMA_simulate2D(maindir, params);

% save the simulated data:
% datatofit = cat(3, predneuralweights, baselinepredneuralweights);
% save('/Volumes/server/Projects/attentionpRF/Simulations/fitData/simulateddata.mat', 'datatofit');
for cond = 1:length(attx0locs)
    params      = [mxecc,attgain,attx0locs(cond),atty0locs(cond),attsd,sigma,suppweight,summweight,spacegrid];
    [stimdrive(:,:,cond), numerator(:,:,cond), suppsurround(:,:,cond), predneuralweights(:,:,cond), predsummedweights(:,:,cond)] ...
        = NMA_simulate2D(maindir, params);
end

% reshape into images

for ii = 1:length(attx0locs)
    numeratorIm(:,:,:,ii) = reshape(stimdrive(:,:,ii), 64, 64, 48);
    suppIm(:,:,:,ii) = reshape(suppsurround(:,:,ii), 64, 64, 48);
    sptPopResp(:,:,:,ii) = reshape(predneuralweights(:,:,ii), 64, 64, 48);
    predsummedweightsIm(:,:,:,ii) = reshape(predsummedweights(:,:,ii), 64, 64, 48);
end


%%%%%% PLOTS -- they are to turn into scripts
cmap = [255,245,240
    254,224,210
    252,187,161
    252,146,114
    251,106,74
    239,59,44
    203,24,29
    165,15,21
    103,0,13]/255;

% Plot the predicted population responses for cued conditions:
clims = [0 30];
for condind = 1:length(attx0locs)
    figure
    subplot(1,4,1)
    imagesc(sum(numeratorIm(:,:,:,condind),3),clims)
    title('Numerator (stimulus drive.*attfield)')
    colormap(cmap)
    axis off
    axis square
    subplot(1,4,2)
    imagesc(sum(suppIm(:,:,:,condind),3),clims)
    title('Suppressive drive')
    colormap(cmap)
    axis off
    axis square
    subplot(1,4,3)
    imagesc(sum(sptPopResp(:,:,:,condind),3),clims)
    title('Population response')
    colormap(cmap)
    set(gcf, 'Position', [0 0 850 220])
    set(gcf,'color','white')
    axis off
    axis square
    subplot(1,4,4)
    imagesc(sum(predsummedweightsIm(:,:,:,condind),3))
    title('Population response')
    colormap(cmap)
    set(gcf, 'Position', [0 0 850 220])
    set(gcf,'color','white')
    axis off
    axis square
end

% Plot the predicted population responses for the baseline/neutral cue condition:
figure
subplot(1,3,1)
imagesc(sum(numeratorIm_base,3),clims)
title('Numerator (stimulus drive.*attfield)')
colormap(cmap)
axis off
subplot(1,3,2)
imagesc(sum(suppIm_base,3),clims)
title('Suppressive drive')
colormap(cmap)
axis off
subplot(1,3,3)
imagesc(sum(baselineSpatialResponse,3),clims)
title('Population response')
colormap(cmap)
set(gcf, 'Position', [0 0 850 220])
set(gcf,'color','white')
axis off


iter=1;
figure;
for ind = 1:200:length(predneuralweights)
    subplot(4,5,iter)
    plot(baselinepredneuralweights(ind,:,1), '--k','LineWidth',2)
    hold on
    plot(predneuralweights(ind,:,1), 'Color',[67, 144, 191]/255,'LineWidth',2)
    hold on
    plot(predneuralweights(ind,:,2), 'Color',[216, 9, 10]/255,'LineWidth',2)
    hold on
    plot(predneuralweights(ind,:,3), 'Color',[112, 173, 71]/255,'LineWidth',2)
    hold on
    plot(predneuralweights(ind,:,4), 'Color',[196, 116, 215]/255,'LineWidth',2)
    if ind == 1
        legend('Neutral','Attend up', 'Attend down', 'Attend left', 'Attend right')
        xlabel('bar position')
        ylabel('estimate of neural activity')
    end
    title(sprintf('Voxel num = %i', ind))
    box off
    set(gcf,'Color','w')
    iter = iter + 1;
end


% pick a good RF, plot its responses across conditions to see the shifts
cmap = [239,243,255;
    189,215,231;
    107,174,214;
    33,113,181]/255;

ind = 519; % hand-picked hehe
figure
set(gcf,'color','w')
% subplot(2,1,1)
plot(baselinepredsummedweights(ind,:), '--k','LineWidth',2)
hold on
plot(predsummedweights(ind,:,1), 'Color',cmap(1,:),'LineWidth',2)
hold on
plot(predsummedweights(ind,:,2), 'Color',cmap(2,:),'LineWidth',2)
hold on
plot(predsummedweights(ind,:,3), 'Color',cmap(3,:),'LineWidth',2)
hold on
plot(predsummedweights(ind,:,4), 'Color',cmap(4,:),'LineWidth',2)
legend('Neutral','AF size = 0.5', 'AF size = 1', 'AF size = 2', 'AF size = 2.5')
xlabel('bar position')
ylabel('estimate of neural activity')
box off
set(gcf, 'Position', [0 0 400 300])

