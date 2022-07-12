function resSumSq = RSSR(params,stim,data)
    rows        = size(stim,1);
    cols        = size(stim,2);
    timepoints  = size(stim,3);
    
    % Compute the time course
    stimVec     = reshape(stim, rows*cols, timepoints);
    %Compute the BOLD response:
    [X,Y] = meshgrid(linspace(-10,10,cols), linspace(-10,10, rows));
    t = (1:timepoints)';
    % Define a hemodynamic impulse response function
    h = t .* exp(-t/5);
    h = h / sum(h); % normalize to sum of 1
    
    [~,RFvec] = createGauissanFields(X,Y,params(1),params(2),params(3));

    pred = conv(stimVec' * RFvec',h);
    pred = pred(1:timepoints,:);
    
    % Calculate residual sum of squares:
    b = mldivide(pred,data);
    resSumSq = sum((data - b*pred).^2);
end 