function popRSS = RSS(currData, attx0, atty0, stim,stim_ecc, stimdrivenRFs, params)
                        
    [~, predsummedweights] = normalizationmodel(stim, stim_ecc, stimdrivenRFs, attx0, atty0,...
         params(1), params(2), params(3));
    resSumSq = zeros(1, length(predsummedweights));
    
for voxel = 1:length(predsummedweights)  
    Y = currData(voxel,:);
    X = predsummedweights(voxel,:);
    X = X';
    Y = Y';
    Xs = [ones(length(X),1) X];
    b = Xs\Y;
    resSumSq(voxel) = sum((Y - b(1) - b(2)*X).^2);
end
   popRSS = mean(resSumSq);
   fprintf('>>>Running voxel %d of %d\n', voxel, length(predsummedweights));
end