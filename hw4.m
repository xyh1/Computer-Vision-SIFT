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

% Utilizing script instead of functions for speed

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
num_butterflies = size(all_butterflies); % number of butterfly images

for fileNum = 1:num_butterflies(1) % apply to all butterfly images
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
num_airplanes = size(all_airplanes); % number of airplane images

for fileNum = 1:num_airplanes(1) % apply to all airplane images
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

[budda_idx, budda_dist] = knnsearch(centers', budda_descriptors');
[butterfly_idx, butterfly_dist] = knnsearch(centers', butterfly_descriptors');
[airplane_idx, airplane_dist] = knnsearch(centers', airplane_descriptors');

budda_threshold = max(budda_dist) - 2*min(budda_dist);
butterfly_threshold = max(butterfly_dist) - 2*min(butterfly_dist);
airplane_threshold = max(airplane_dist) - 2*min(airplane_dist);

useless_budda = find(budda_dist > budda_threshold);
useless_butterfly = find(butterfly_dist > butterfly_threshold);
useless_airplane = find(airplane_dist > airplane_threshold);

for useless = useless_budda
	budda_idx(useless) = [];
	budda_dist(useless) = [];
end

for useless = useless_butterfly
	butterfly_idx(useless) = [];
	butterfly_dist(useless) = [];
end

for useless = useless_airplane
	airplane_idx(useless) = [];
	airplane_dist(useless) = [];
end

budda_hist = hist(budda_idx, num_clustors);
butterfly_hist = hist(butterfly_idx, num_clustors);
airplane_hist = hist(airplane_idx, num_clustors);

% Normalize bin counts by total number of SIFT features binned for that particular class
budda_histN = budda_hist./max(budda_hist);
butterfly_histN = butterfly_hist./max(butterfly_hist);
airplane_histN = airplane_hist./max(airplane_hist);

%% Find SIFT Features for Test Images
rtest_dir1 = 'midterm_data/midterm_data_reduced/TestDataset_1/';
rtest_dir2 = 'midterm_data/midterm_data_reduced/TestDataset_2/';
rtest_dir3 = 'midterm_data/midterm_data_reduced/TestDataset_3/';

rtest1_struct = struct();
rtest2_struct = struct();
rtest3_struct = struct();

% Get all .jpg images from directory
all_tdir1 = dir(strcat(rtest_dir1,'*.jpg'));
all_tdir2 = dir(strcat(rtest_dir2,'*.jpg'));
all_tdir3 = dir(strcat(rtest_dir2,'*.jpg'));

% number of test images
num_tdir1 = size(all_tdir1); 
num_tdir2 = size(all_tdir2); 
num_tdir3 = size(all_tdir3); 
%total_tdir_num = num_tdir1(1)+num_tdir2(1)+num_tdir3(1);

for fileNum = 1:num_tdir1(1) % apply to all rtest1 images
	filename = all_tdir1(fileNum).name;
	tdir1_image = imread(strcat(rtest_dir1,filename));

	if size(tdir1_image, 3) > 1 % make sure image is not already gray
		tdir1_imageG = im2single(rgb2gray(tdir1_image));
	end

	[f,d] = vl_sift(tdir1_imageG);
	%filename = strcat('testdir1_',regexprep(filename,'.jpg',''));
	rtest1_struct(fileNum).name = filename;
	%rtest1_struct(fileNum).f = f;
	rtest1_struct(fileNum).d = d;
end

for fileNum = 1:num_tdir2(1) % apply to all rtest2 images
	filename = all_tdir2(fileNum).name;
	tdir2_image = imread(strcat(rtest_dir1,filename));

	if size(tdir2_image, 3) > 1 % make sure image is not already gray
		tdir2_imageG = im2single(rgb2gray(tdir2_image));
	end

	[f,d] = vl_sift(tdir2_imageG);
	%filename = strcat('testdir2_',regexprep(filename,'.jpg',''));
	rtest2_struct(fileNum).name = filename;
	%rtest2_struct(fileNum).f = f;
	rtest2_struct(fileNum).d = d;
end

for fileNum = 1:num_tdir3(1) % apply to all rtest3 images
	filename = all_tdir3(fileNum).name;
	tdir3_image = imread(strcat(rtest_dir1,filename));

	if size(tdir3_image, 3) > 1 % make sure image is not already gray
		tdir3_imageG = im2single(rgb2gray(tdir3_image));
	end

	[f,d] = vl_sift(tdir3_imageG);
	%filename = strcat('testdir3_',regexprep(filename,'.jpg',''));
	rtest3_struct(fileNum).name = filename;
	%rtest3_struct(fileNum).f = f;
	rtest3_struct(fileNum).d = d;
end