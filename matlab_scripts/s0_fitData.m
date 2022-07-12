clear 
maindir = '/Volumes/server/Projects/attentionpRF/Simulations/matlab_scripts';
addpath(genpath(maindir));
% get the stim and data directories:
stimfiles = fullfile(maindir, 'stimfiles','stim.mat');
datafiles = fullfile(maindir,'fitData','simulateddata_subset.mat');    

% load the stimulus files and the simulated dataset
load(stimfiles)
stim = stim(:,:,1:end-1);
load(datafiles)

% define the max eccentricity
stim_ecc = 12;
attentionLocations = [1, 0, 5;
    2, 0, -5;
    3, -5, 0;
    4, 5, 0;
    5, 0, 0];

estimatedParameters = NMA_fit(datatofit_subset, stim, stim_ecc, attentionLocations);