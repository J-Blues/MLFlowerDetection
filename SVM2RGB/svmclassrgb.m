%% SVM RGB:

%% Select one RGB image
[FileNames,PathName] = uigetfile('*.jfif', 'Select RGB image:'); %select the file
img = imread(fullfile(PathName,FileNames));%load it to matlab
im_height = size(img,1);%array size no bands
im_width  = size(img,2);

%% RGB Load Each Band and Convert to vector
red_layer   = img(:,:,1); %read each band for RGB
green_layer = img(:,:,2);
blue_layer  = img(:,:,3);
red_layer   = red_layer(:); %convert each band to 1-D array
green_layer = green_layer(:);
blue_layer  = blue_layer(:);
img_vector  = [red_layer green_layer blue_layer]; %create a list
size(blue_layer)

%% Select ROI by 1 pixel
figure(1); imshow(img); title('Select background');

[xBack, yBack] = ginput(1); title('Select foreground');
[xFront, yFront] = ginput(1); % Identify axes coordinates

xBack  = round(xBack);        % Round to nearest decimal or integer
yBack  = round(yBack);
xFront = round(xFront);
yFront = round(yFront);
close(1);

%% Read image coordinates
background = img(xBack(1):xBack(1),yBack(1):yBack(1),:);
foreground = img(xFront(1):xFront(1),yFront(1):yFront(1),:);

%% Read image coordinates for the background Pixel
bg_red = background(:,:,1); 
bg_red = bg_red(:);
bg_green = background(:,:,2);
bg_green = bg_green(:);
bg_blue = background(:,:,3);
bg_blue = bg_blue(:);

%% Read image coordinates for the foreground Pixel
fg_red = foreground(:,:,1);
fg_red = fg_red(:);
fg_green = foreground(:,:,2);
fg_green = fg_green(:);
fg_blue = foreground(:,:,3);
fg_blue = fg_blue(:);

%% New Vectors
bg_feats = [bg_red bg_green bg_blue];
fg_feats = [fg_red fg_green fg_blue];

%% Identify spectral features based on selected ROI 
features = [bg_feats; fg_feats];
groups   = [-ones(length(bg_red),1); ones(length(fg_red),1)]; % Identify label 
options = optimset('maxiter',1000); % Run up to 1000 times and find the vector

%% Train support vector machine (SVM) classifier for one-class and binary classification
svm_struct = fitcsvm(double(features), groups);                                           
cimg = predict(svm_struct,double(img_vector));
cimg = reshape(cimg, im_height, im_width);

figure, 
subplot(1,2,1),imagesc(img),title('RGB image')
subplot(1,2,2),imagesc(cimg),title('SVM based classification')

%% Select a test image 
[FileNames1,PathName1] = uigetfile('*.jfif', 'Select test RGB image:'); %select the file
img1 = imread(fullfile(PathName1,FileNames1));%load it to matlab
im_height1 = size(img1,1);%array size no bands
im_width1  = size(img1,2);

red_layer1   = img1(:,:,1); %read each band for RGB
green_layer1 = img1(:,:,2);
blue_layer1  = img1(:,:,3);
red_layer1   = red_layer1(:); %convert each band to 1-D array
green_layer1 = green_layer1(:);
blue_layer1  = blue_layer1(:);
img_vector1  = [red_layer1 green_layer1 blue_layer1]; %create a list

cimg1 = predict(svm_struct,double(img_vector1));
cimg1 = reshape(cimg1, im_height1, im_width1);

figure, 
subplot(1,2,1),imagesc(img1),title('Test RGB image')
subplot(1,2,2),imagesc(cimg1),title('Trained SVM based classification')
