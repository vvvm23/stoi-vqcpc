clear all, close all, clc
addpath(genpath(fullfile(pwd, 'functions')))

% Directory where you downloaded the data
% should be the same as the variable 'data_src' on top of download.sh
corpora_dir = '/path/to/downloaded/corpora';

% Directory where you want to save the  sets of generated binaural files:
output_dir  = '/path/to/created/binaural/signals';

% Directory containg the .csv files with the needed information
csv_dir = fullfile(pwd, 'csv');

sets    = {'train', 'dev', 'test_iso', 'test_real'};
for set_ind = 1:length(sets)
    sub_set = sets{set_ind};
    generate_set(csv_dir, sub_set, corpora_dir, output_dir);
end
