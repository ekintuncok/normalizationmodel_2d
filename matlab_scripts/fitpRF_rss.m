function rss = fitpRF_rss(currData, stim, spacegrid, rfparams)

[inputStim, X, Y] = createSpatialGrid(stim, stim_ecc, spacegrid);

[~,RFvec] = createRFs(X,Y,rfparams(1),rfparams(2),rfparams(3));

% vectorize the stim input:
% prediction is the product of the stim matrix with RFs
pred = RFvec*inputStim;

% get the beta vals for regions of interest
b = mldivide(pred',currData');
rss = sum((currData - b*pred).^2);

end

