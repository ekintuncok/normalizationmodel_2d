%% set the parameters
npoints     = params(2);
mxecc       = params(3);
RFsd        = params(4);
attgain     = params(5);
attx0       = params(6);
attsd       = params(7);
summationsd = params(8);
sigma       = params(9);
visualize   = params(10);

%% simulate
[X,Y] = meshgrid(linspace(-mxecc,mxecc,npoints), linspace(-mxecc,mxecc, npoints));
RF = exp(-((X-0).^2 +(Y-0).^2)./(2*RFsd).^2);
stim = zeros(npoints,npoints);
stim(:,450:500) = 1;
attfield = exp(-((X-attx0).^2 +(Y-attx0).^2)./(2*attsd).^2);
attpRF = RF .* attfield;
popresponse = conv2(attpRF, stim,'same');

fH = figure; clf;
x = linspace(-mxecc,mxecc, npoints);
lw = 3;
fs = 10;
subplot(1,5,1);
imagesc(x,x,RF);
title('RF center');
plotOptions(gca, fs);
subplot(1,5,2);
imagesc(x,x,attfield);
title('Attention field');
plotOptions(gca, fs);
subplot(1,5,3);
imagesc(x,x,stim)
title('Stimuli');
plotOptions(gca, fs);
subplot(1,5,4);
imagesc(x, x, attpRF);
title('pRF under the attentional load');
subplot(1,5,5);
imagesc(x,x,popresponse)
title('Population response');
plotOptions(gca, fs);
set(gcf,'Position',[0 0 1100 450])