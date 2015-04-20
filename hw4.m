close all
clear all
format long
clc

%% Initialize environment by enabling vlfeat
current_pwd = pwd; % Get working directory
clear_pwd = '/storage-home/x/xyh1';
if (strcmp(current_pwd(1:20),clear_pwd) == 1)
	run('~/src/vlfeat-0.9.20/toolbox/vl_setup');
	sprintf('Success; imported vlfeat to MATLAB!\n')
	vl_version verbose
% TODO else if windows 
else
	sprintf('Error; could not utilize VLFeat!\n')
end

%% Buddas from Reduced Training Dataset
budda_dir = 'midterm_data/midterm_data_reduced/TrainingDataset/022.buddha-101/';
budda_struct = struct(); % Get all .jpg images from directory
all_buddas = dir(strcat(budda_dir,'*.jpg'));
num_buddas = size(all_buddas); % number of budda images

for fileNum = 1:num_buddas(1) % apply to all budda images
	filename = all_buddas(fileNum).name;
	budda_image = imread(strcat(budda_dir,filename));

	if size(budda_image, 3) > 1 % make sure image is not already gray
		budda_imageG = im2single(rgb2gray(budda_image));
	end

	[f,d] = vl_sift(budda_imageG); % frames, descriptors
	%filename = strcat('budda_',regexprep(filename,'.jpg',''));
	budda_struct(fileNum).name = filename;
	%budda_struct(fileNum).f = f;
	budda_struct(fileNum).d = d;
end

budda_descriptors = horzcat(budda_struct.d); % concatenate all descriptors

%% Butterfly from Reduced Training Dataset
butterfly_dir = 'midterm_data/midterm_data_reduced/TrainingDataset/024.butterfly/';
butterfly_struct = struct(); 
all_butterflies = dir(strcat(butterfly_dir,'*.jpg')); % Get all .jpg images from directory
num_butterflies = size(all_butterflies); % number of budda images

for fileNum = 1:num_butterflies(1) % apply to all budda images
	filename = all_butterflies(fileNum).name;
	butterfly_image = imread(strcat(butterfly_dir,filename));

	if size(butterfly_image, 3) > 1 % make sure image is not already gray
		butterfly_imageG = im2single(rgb2gray(butterfly_image));
	end

	[f,d] = vl_sift(butterfly_imageG);
	%filename = strcat('butterfly_',regexprep(filename,'.jpg',''));
	butterfly_struct(fileNum).name = filename;
	%butterfly_struct(fileNum).f = f;
	butterfly_struct(fileNum).d = d;
end

butterfly_descriptors = horzcat(butterfly_struct.d); % concatenate all descriptors

%% Airplane from Reduced Training Dataset
airplane_dir = 'midterm_data/midterm_data_reduced/TrainingDataset/251.airplanes/';
airplane_struct = struct();
all_airplanes = dir(strcat(airplane_dir,'*.jpg')); % Get all .jpg images from directory
num_airplanes = size(all_airplanes); % number of budda images

for fileNum = 1:num_airplanes(1) % apply to all budda images
	filename = all_airplanes(fileNum).name;
	airplane_image = imread(strcat(airplane_dir,filename));

	if size(airplane_image, 3) > 1 % make sure image is not already gray
		airplane_imageG = im2single(rgb2gray(airplane_image));
	end

	[f,d] = vl_sift(airplane_imageG);
	%filename = strcat('airplane_',regexprep(filename,'.jpg',''));
	airplane_struct(fileNum).name = filename;
	%airplane_struct(fileNum).f = f;
	airplane_struct(fileNum).d = d;
end

airplane_descriptors = horzcat(airplane_struct.d); % concatenate all descriptors

%% Putting Training Data together
all_descriptors = horzcat(budda_descriptors,butterfly_descriptors,airplane_descriptors);
num_clustors = 1000;

%[centers, assignments] = vl_kmeans(single(all_descriptors), num_clustors);

% kmeans++ 
% http://en.wikipedia.org/wiki/K-means%2B%2B
[centers, assignments] = vl_kmeans(single(all_descriptors), num_clustors, 'Initialization', 'plusplus');

[budda_idx, budda_dist] = knnsearch(centers, budda_descriptors);
[butterfly_idx, butterfly_dist] = knnsearch(centers, butterfly_descriptors);
[airplane_idx, airplane_dist] = knnsearch(centers, airplane_descriptors);

budda_hist = hist(budda_idx, num_clustors);
butterfly_hist = hist(butterfly_idx, num_clustors);
airplane_hist = hist(airplane_idx, num_clustors);