function estimated_rfparams = getSensoryDrivenpRF(baselineactivity, stim)

spacegrid = sqrt(length(baselineactivity));
x0 = [0 , 0, 1];
lb = [-10, -10,  0.01];
ub = [ 10,  10, 10];    

estimated_rfparams = zeros(3,length(baselineactivity));
for rf_idx = 1:length(baselineactivity)
    fprintf('Solving pRF for voxel %d of %d\n', rf_idx, length(baselineactivity));
    currData = baselineactivity(rf_idx,:);
    funtomin = (@(rfparams) fitpRF_rss(currData, stim, spacegrid, rfparams));
    estimated_rfparams(:,rf_idx) = fmincon(funtomin, x0, [],[],[],[],lb,ub);
end

end