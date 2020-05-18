%% SVM Gray Scale:

%%   Open file selection dialog box with Multiselect 'on'(Multiple node selection)
[FileNames,PathName] = uigetfile('*.jfif', 'Chose class1 images:','MultiSelect','on');
nfiles = length(FileNames); %returns the length of the largest array dimension
image_array = [];

%% For Loop for Iris Folder and Sunflower Folder: 
%   Runs into images - Converts RGB image to grayscale, 
%   Resize images to the same size,
%   Reshape 
for i = 1:nfiles
   im = imread(fullfile(PathName,FileNames{i}));
   im = double(rgb2gray(im));
   im = imresize(im,[150 150]);
   im = reshape(im,size(im,1)*size(im,2),1);
   image_array(:,i) = im;
end
  
[FileNames2,PathName2] = uigetfile('*.jfif', 'Chose class2 images:','MultiSelect','on');
nfiles2 = length(FileNames2); 
image_array2 = [];
for i = 1:nfiles2
   im = imread(fullfile(PathName2,FileNames2{i}));
   im = double(rgb2gray(im));
   im = imresize(im,[150 150]);
   im = reshape(im,size(im,1)*size(im,2),1);
   image_array2(:,i) = im;
end

%% Create array of all ones and associate to Tag name
class1 = ones(1,nfiles);
class2 = ones(1,nfiles2)*2;

%% TrainData 
traindata = [image_array image_array2];
classdata = [class1 class2];

%% Try 1: Train support vector machine (SVM) classifier for one-class and binary classification
%  Kernel scale parameter, specified as a positive scalar.

SVMStruct = fitcsvm (traindata', classdata,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto','CrossVal','on');
    
%% Try 2: Train support vector machine (SVM) classifier for one-class and binary classification
%  Estimate loss using cross-validation
%SVMStruct = fitcsvm (traindata', classdata,'CrossVal','on');

% Try 1: Gen Error for 22 pair of images 0.318181818181818 - 30% 
% Try 2: Gen Error for 22 pair of images 0.522727272727273 - 50% Error Bad result
genError = kfoldLoss(SVMStruct);