inputs = IrisData';
targets = IrisLable';

%% Tan-sigmoid transfer function in the hidden layer and linear transfer function in the output layer
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize); 

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%% Train shallow neural network
[net, tr] = train(net, inputs, targets);

outputs = net(inputs);
errors = gsubtract(targets, outputs);
performance = perform(net, targets, outputs);

%% Select the file (One Image)
[FileNames,PathName] = uigetfile('*.jfif', 'Select RGB image:'); 
img = imread(fullfile(PathName,FileNames));%load it to matlab
im_height = size(img,1);%array size no bands
im_width  = size(img,2);

%% RGB read each band
red_layer   = img(:,:,1); 
green_layer = img(:,:,2);
blue_layer  = img(:,:,3);

%% Convert each band to 1-D array
red_layer   = red_layer(:); %
green_layer = green_layer(:);
blue_layer  = blue_layer(:);

%% Create a list
img_vector  = [red_layer green_layer blue_layer];

annres = net(double(img_vector'));
annres1 = annres';
annres_img_noflow = reshape(annres1(:,1), im_height, im_width);
annres_img_flow = reshape(annres1(:,2), im_height, im_width);

figure, 
subplot(1,3,1),imagesc(img),title('RGB image')
subplot(1,3,2),imagesc(annres_img_flow),title('ANN based flower class')
subplot(1,3,3),imagesc(annres_img_noflow),title('ANN based no flower class')
