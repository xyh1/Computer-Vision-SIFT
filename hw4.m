close all
clear all
format long
clc

current_pwd = pwd; % Get working directory
clear_pwd = '/storage-home/x/xyh1';
if (strcmp(current_pwd(1:20),clear_pwd) == 1)
	run('~/src/vlfeat-0.9.20/toolbox/vl_setup');
	'Success; imported vlfeat to MATLAB!\n'
	vl_version verbose
% TODO else if windows 
else
	'Error; could not utilize VLFeat!\n'
end

% Get all .jpg images from directory
dir('midterm_data/midterm_data_reduced/TrainingDataset/022.buddha-101/*.jpg')