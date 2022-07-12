% NMA in receptive field shifts
% ET & JW 02/2022
% Simulates receptive field position and amplitude data based on
% Normalization Model of Attention
clear all
close all
setDefinitions = {'Baseline','Variable RF size','Variable AF gain', 'Variable AF size'};

npoints     = 1001;
mxecc       = 10;
summationsd = 2;
sigma       = 0.1;
visualize   = 0;
attx0       = 0;
atty0       = 0;
% Condition indices: Code for which set is manipulated. This makes the
% indexing and plotting with the accurate legend way easier.
numSets = 4;
baseline = 1; RFsize   = 2; AFgain   = 3;AFsize   = 4;

% Simulate baseline data: Set the attention gain parameter to close to
% zero for an 'average' RF size:
RFsd{1}        = 2;
RFsd{2}        = 1:5;

attgain{1}     = 0.01;
attgain{2}     = 1.5;
attgain{3}     = 1:5;

attsd{1}       = 2;
attsd{2}       = 1:5;
% We will be using the simulated baseline data with close to zero AF gain
% to compare the changes in model predictions:
params = [combvec(ones,npoints,mxecc,RFsd{1},attgain{1},attx0,atty0,attsd{1},summationsd,sigma,0)';...
    combvec(RFsize*ones,npoints,mxecc,RFsd{2},attgain{2},attx0,atty0,attsd{1},summationsd,sigma,0)';...
    combvec(AFgain*ones,npoints,mxecc,RFsd{1},attgain{3},attx0,atty0,attsd{1},summationsd,sigma,0)';...
    combvec(AFsize*ones,npoints,mxecc,RFsd{1},attgain{2},attx0,atty0,attsd{2},summationsd,sigma,0)'];

RFsizeColidx = 4; AFgainColidx = 5; AFsizeColidx = 7;
% popresponse = zeros(length(npoints),length(npoints), size(params,1),numSets);
populationResponse = cell(numSets,1);

for caseidx = 1:numSets
    currCase = params(params(:,1) == caseidx,:,:);
    for subsetPrm = 1:size(currCase,1)
        currSubset = currCase(subsetPrm,:);
        [x, populationResponse{caseidx}(:,:,subsetPrm), ~] = NMA_simulate1D(currSubset);
    end
end
%%
% pick the locations around the AF (close the far)
locs  = -6:6;
stimidx = zeros(size(locs));
x = linspace(-mxecc,mxecc, npoints);
for ii = 1:length(locs)
    k = locs(ii);
    [~, stimidxX(ii)] = min(abs(x-(attx0+k)));
    [~, stimidxY(ii)] = min(abs(x-(atty0+k)));
end

% Baseline vals (we will not be looping through them). N and P stand for
% the neural and population responses

[amp_p_baseline, center_p_baseline] = max(populationResponse{1}(:, stimidx, :));
[amp_n_baseline, center_n_baseline] = max(populationResponse{1}(stimidx, :,:),[],2);
for setInd = 2:length(populationResponse)
    tag = setDefinitions{setInd};
    [resp_pop, respidx_pop] = max(populationResponse{setInd}(:, stimidx, :));
    [resp_neural, respidx_neural] = max(populationResponse{setInd}(stimidx, :,:),[],2);
    % Extract the manipulated params
    paramstoVis = params(params(:,1) == setInd,:);
    cmap = brewermap(size(paramstoVis,1),'Reds');
    fs = 16;
    Legend = cell(size(paramstoVis,1)+1,1);
    Legend{1} = 'Baseline (sensory)';
    
    % set the info on fixed params
    if setInd == 2 % this is when RF size is varied
        myLabels = {sprintf('AF gain = %g, AF size = %g', paramstoVis(1,AFgainColidx), paramstoVis(1,AFsizeColidx))}; %variable RF size
        for iter=1:size(paramstoVis,1)
            Legend{iter+1}=sprintf('RF size = %i', paramstoVis(iter,RFsizeColidx));
        end
    elseif setInd == 3
        myLabels = {sprintf('RF size = %g, AF size = %g', paramstoVis(1,RFsizeColidx), paramstoVis(1,AFsizeColidx))}; %variable AF gain
        for iter=1:size(paramstoVis,1)
            Legend{iter+1}=sprintf('AF gain = %i', paramstoVis(iter,AFgainColidx));
        end
    else
        myLabels = {sprintf('RF size = %g, AF gain = %g', paramstoVis(1,RFsizeColidx), paramstoVis(1,AFgainColidx))}; %variable AF size
        for iter=1:size(paramstoVis,1)
            Legend{iter+1}=sprintf('AF size = %i', paramstoVis(iter,AFsizeColidx));
        end
    end
    for currSet = 1:size(paramstoVis,1)
        % Amplitude plots
        resp_pop = squeeze(resp_pop);
        resp_neural = squeeze(resp_neural);
        figure(1);
        subplot(2,3,setInd-1)
        plot(x(stimidx),amp_p_baseline/norm(amp_p_baseline), 'k', 'LineWidth',2)
        hold on
        plot(x(stimidx),resp_pop/norm(resp_pop),'LineWidth',2)
        set(gca,'ColorOrder',cmap)
        xlabel('Stimulus Center')
        title('Response amplitude for different stimulus centers')
        plotOptions(gca, fs);
        xline(0,'--')
        lgd = legend(Legend, 'Location', 'southeast');
        lgd.FontSize = 11;
        hold on
        
        subplot(2,3,setInd+2)
        plot(x(stimidx),amp_n_baseline/norm(amp_n_baseline), 'k', 'LineWidth',2)
        hold on
        plot(x(stimidx),resp_neural/norm(resp_neural),'LineWidth',2)
        set(gca,'ColorOrder',cmap)
        xlabel('Stimulus Center')
        title('Response amplitude of different sensory RFs')
        set(gcf,'Position',[0 0 800 600])
        xline(0,'--')
        plotOptions(gca, fs);
        hold on
    end
    
    % Shift plots:
    figure(2);
    sgtitle('Model predictions for varying 1) RF size, 2) AF gain and 3) AF size', 'FontSize',20)
    for currSet = 1:size(paramstoVis,1)
        subplot(2,3,setInd-1)
        if currSet == 1
            scatter(x(center_p_baseline),x(stimidx),80,'MarkerEdgeColor',[0 0 0],...
                'MarkerFaceColor',[0 0 0],'MarkerFaceAlpha',.01)
        end
        text(-3, 6, sprintf('%s\n%s\n%s', myLabels{1}), ...
            'FontWeight', 'bold', 'horizontalalignment', 'center', 'verticalalignment', 'top');
        ylim([locs(1), locs(end)])
        hline  = refline(1);
        hline.Color = 'k';
        hline.Annotation.LegendInformation.IconDisplayStyle = 'off';
        attField = xline(attx0, '--','Color','k','LineWidth',2);
        attField.Annotation.LegendInformation.IconDisplayStyle = 'off';
        hold on
        scatter(x(respidx_pop(:,:,currSet)),x(stimidx),40,'LineWidth',2)
        ax1 = gca;
        set(ax1,'ColorOrder',cmap)
        if setInd == 2
            xlabel('Preferred position in the population response')
            ylabel('Stimulus center')
        end
        lgd = legend(Legend, 'Location', 'southeast');
        lgd.FontSize = 11;
        title(tag)
        hold on
        grid(ax1, 'on')
        set(ax1,'Layer','top','GridColor','k','GridAlpha',1,'GridLineStyle','--','GridAlpha',0.2);
        set(ax1,'XTick', locs(1):1:locs(end))
        plotOptions(ax1, fs);
        subplot(2,3,setInd+2)
        if currSet == 1
            scatter(x(center_n_baseline),x(stimidx),80,'MarkerEdgeColor',[0 0 0],...
                'MarkerFaceColor',[0 0 0],'MarkerFaceAlpha',.01)
        end
        myText = text(-3, 6, sprintf('%s\n%s\n%s', myLabels{1}), ...
            'FontWeight', 'bold', 'horizontalalignment', 'center', 'verticalalignment', 'top');
        ylim([locs(1), locs(end)])
        hline  = refline(1);
        hline.Color = 'k';
        hold on
        scatter(x(respidx_neural(:,:,currSet)),x(stimidx),'LineWidth',2)
        ax2 = gca;
        set(gca,'ColorOrder',cmap)
        if setInd == 2
            xlabel('Attentional RF center')
            ylabel('Sensory RF center')
        end
        xline(attx0, '--','Color','k','LineWidth',2)
        hold on
        grid(ax2, 'on')
        set(ax2,'Layer','top','GridColor','k','GridAlpha',1,'GridLineStyle','--','GridAlpha',0.2);
        set(ax2,'XTick', locs(1):1:locs(end))
        plotOptions(ax2, fs);
    end
end
% produces the amplitude plots based on the population response output from the
% model. To plot the neural response, extracts the stimulus centers of
% interest from the rows of the population response output. To plot the
% population response, extracts the stim centers of interest from the
% columns of the population response output. Since the stim images are
% are centered in the rows and are convolved with the RFs row by row, the
% column slice returns the response from the neural image for a given
% stimulus center.



