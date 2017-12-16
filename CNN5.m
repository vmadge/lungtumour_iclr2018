% Trains a custom CNN based on the datastore object in datastore.mat

close all; clear; clc;
t1 = tic; % Start timing

% Load the dataset
load('datastore.mat');

% Define the CNN architecture as proposed in the paper
% The first layer is downsampled to 64x64
layers = [imageInputLayer([64 64 1],'Normalization','none','Name','inputl')
          convolution2dLayer([10 10],64,'Stride',1,'Padding',5,'Name','conv1')
          reluLayer('Name','relu1')
          maxPooling2dLayer(3,'Name','max1') 
          dropoutLayer(0.1,'Name','dropout1')
          convolution2dLayer([5 5],192,'Stride',1,'Padding',5,'Name','conv2')
          reluLayer('Name','relu2')
          maxPooling2dLayer(3,'Name','max2') 
          dropoutLayer(0.1,'Name','dropout2')
          convolution2dLayer([5 5],384,'Stride',1,'Name','conv3')
          reluLayer('Name','relu3')
          convolution2dLayer([3 3],256,'Stride',1,'Name','conv4')
          reluLayer('Name','relu4')
          convolution2dLayer([3 3],256,'Stride',1,'Name','conv5')
          reluLayer('Name','relu5')
          convolution2dLayer([3 3],256,'Stride',1,'Name','conv6')
          reluLayer('Name','relu6')
          convolution2dLayer([3 3],128,'Stride',1,'Name','conv7')
          reluLayer('Name','relu7')
          maxPooling2dLayer(3,'Name','max3') 
          dropoutLayer(0.5,'Name','dropout3')
          fullyConnectedLayer(2,'Name','full2')
          softmaxLayer('Name','softm')
          classificationLayer('Name','out')];

% Resize images to the input layer dimensions as they are read in
inputSize = layers(1).InputSize(1:2);
imds.ReadFcn = @(loc)imresize(imread (loc), inputSize);

options = trainingOptions('sgdm','MaxEpochs',20,...
    'InitialLearnRate',0.001);

% Defined stratified subsets 
[subset{1} subset{2} subset{3} subset{4} subset{5} ...
    subset{6} subset{7} subset{8} subset{9} subset{10}] = ...
    splitEachLabel(imds, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);

% For each subset
for i=1:10
    fprintf('Subset %i:\n', i);
    % Split subset into random train and test datasets
    [train{i}, test{i}] = splitEachLabel(subset{i}, 0.7, 'randomized');
    % Train the CNN
    convnet = trainNetwork(train{i},layers,options);
    % Predict labels for the test data
    prediction{i} = classify(convnet, test{i});
end

% Save teh results to be later analyzed.
save('cnn.mat');
t2 = toc(t1);
fprintf('Success: %i minutes to compute.\n', floor(t2/60));