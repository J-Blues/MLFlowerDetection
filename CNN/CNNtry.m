[FileNames,PathName] = uigetfile('*.jfif', 'Chose class1 images:','MultiSelect','on');
nfiles = length(FileNames); 
image_array = [];

%% Qube
for i = 1:nfiles
   im = imread(fullfile(PathName,FileNames{i}));
   im_r = imresize(im(:,:,1),[227 227]);
   im_g = imresize(im(:,:,2),[227 227]);
   im_b = imresize(im(:,:,3),[227 227]);
   im = cat(3,im_r,im_g,im_b);
   %im = reshape(im,size(im,1)*size(im,2),1);
   image_array(:,:,:,i) = im;
end

%% Annotation (1 - Irus, 2 - Sunflower)
targets = categorical([1;1;1;2;2;2]);

%% AlexNet minus 3 last layers
net = alexnet;
layersTransfer = net.Layers(1:end-3);
inputSize = net.Layers(1).InputSize;

% Only two: Irus + Sunflower
numClasses = 2;

%% 
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% Original pixel range of image
pixelRange = [0 255];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),image_array, ...
    'DataAugmentation',imageAugmenter);

%% Classification function (number of itterations...)
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress');

%% Network Training
netTransfer = trainNetwork(image_array,targets',layers,options);

[FileNames,PathName] = uigetfile('*.jfif', 'Chose class1 images:','MultiSelect','on');
nfiles = length(FileNames); 
image_test = [];
for i = 1:nfiles
   im = imread(fullfile(PathName,FileNames{i}));
   im_r = imresize(im(:,:,1),[227 227]);
   im_g = imresize(im(:,:,2),[227 227]);
   im_b = imresize(im(:,:,3),[227 227]);
   im = cat(3,im_r,im_g,im_b);
   %im = reshape(im,size(im,1)*size(im,2),1);
   image_test(:,:,:,i) = im;
end

predictedLabels = classify(netTransfer,image_test)';
