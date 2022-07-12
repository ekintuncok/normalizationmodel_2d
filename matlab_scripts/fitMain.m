function estimatedParameters = fitMain(data, stimdrivenRFs,stim, stim_ecc, attentionLocations)

x0 = [0.5 , 0.1 , 0.01]; 
lb = [0, 0,  0]; % Lower bound
ub = [5, 5, 2];    % Lower bound
for ii = 1:length(attentionLocations)
    currData = data(:,:,ii);
    attx0 = attentionLocations(ii,2);
    atty0 = attentionLocations(ii,3);
    fun = @(p) RSS(currData, attx0, atty0, stim,stim_ecc, stimdrivenRFs, p);
    estimatedParameters(:,ii) = fmincon(fun,x0,[],[],[],[],lb,ub);
end
end