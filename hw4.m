close all
clear all
format long
clc

%% Initialize environment by enabling vlfeat
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

%% Buddas from Reduced Training Dataset
budda_dir = 'midterm_data/midterm_data_reduced/TrainingDataset/022.buddha-101/';
budda_struct = struct();
% Get all .jpg images from directory
all_buddas = dir(strcat(budda_dir,'*.jpg'));
num_buddas = size(all_buddas); % number of budda images

for fileNum = 1:num_buddas(1) % apply to all budda images
	filename = all_buddas(fileNum).name;
	budda_image = imread(strcat(budda_dir,filename));
	if size(budda_image, 3) > 1 % make sure image is not already gray
		budda_imageG = im2single(rgb2gray(budda_image));
	end
	[f,d] = vl_sift(budda_imageG); % frames, descriptors
	pattern = '.jpg';
	replacement = '';
	filename = regexprep(filename,pattern,replacement);
	budda_struct.(filename) = d;
end

%% Butterfly from Reduced Training Dataset
butterfly_dir = 'midterm_data/midterm_data_reduced/TrainingDataset/022.buddha-101/';
butterfly_struct = struct();
% Get all .jpg images from directory
all_butterflies = dir(strcat(butterfly_dir,'*.jpg'));
num_butterflies = size(all_butterflies); % number of budda images

for fileNum = 1:num_butterflies(1) % apply to all budda images
	filename = all_butterflies(fileNum).name;
	butterfly_image = im2single(rgb2gray(imread(strcat(butterfly_dir,filename))));
	if size(butterfly_image, 3) > 1 % make sure image is not already gray
		butterfly_imageG = im2single(rgb2gray(butterfly_image));
	end
	[f,d] = vl_sift(butterfly_image); % frames, descriptors
	[pathstr,name,ext] = fileparts(filename)
	butterfly_struct.(name) = d;
end