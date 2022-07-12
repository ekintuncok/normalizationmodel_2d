function NMA_fig1(popresp,summation)
fH2 = figure;
set(fH2, 'Color', 'w');
locs  = -5:5;
stimidx = zeros(size(locs));
for ii = 1:length(locs)
    k = locs(ii);
    [~, stimidx(ii)] = min(abs(x-(attx0+k)));
end

subplot(1,3,1);
plot(x, popresp(:, stimidx), 'LineWidth', lw);
title('Population response');
plotOptions(gca, fs);
set(gca, 'XTick', x(stimidx),  'XGrid', 'on', 'GridAlpha', 1)
hold on
[~, respidx_pop] = max(popresp(:, stimidx));yl = get(gca, 'YLim');
plot(x(respidx_pop)*[1 1], yl, 'r--')
subplot(1,3,2);
plot(x, popresp(stimidx,:), 'LineWidth', lw)
title('Neural response');
plotOptions(gca, fs);
hold on; set(gca, 'XTick', x(stimidx),  'XGrid', 'on', 'GridAlpha', 1)
[~, respidx_neural] = max(popresp(stimidx, :),[],2); yl = get(gca, 'YLim');
plot(x(respidx_neural)*[1 1], yl, 'r--')

subplot(1,3,3);
plot(x, summation(stimidx,:), 'LineWidth', lw)
title('Spatial summation');
plotOptions(gca, fs);
hold on; set(gca, 'XTick', x(stimidx),  'XGrid', 'on', 'GridAlpha', 1)
[~, respidx] = max(summation(stimidx, :),[],2); yl = get(gca, 'YLim');
plot(x(respidx)*[1 1], yl, 'r--')
set(gcf,'Position',[0 0 1200 340])
end