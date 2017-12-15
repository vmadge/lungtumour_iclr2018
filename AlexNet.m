% Retrains an AlexNet based on the datastore object in datastore.mat

close all; clear; clc;

% Begin timing the operation
t1 = tic;

% Load the dataset
load('datastore.mat');

% load the pretrained alexnet
net = alexnet;

% Transfer the layers, and redefine the output for only 2 classes
layers = net.Layers(1:end-3);
layers(end+1) = fullyConnectedLayer(2,... 
    'WeightLearnRateFactor',10,... 
    'BiasLearnRateFactor',20); 
layers(end+1) = softmaxLayer(); 
layers(end+1) = classificationLayer();

% Resize each image to be the dimensions of the input layer
inputSize = net.Layers(1).InputSize(1:2);
imds.ReadFcn = @(loc)cat(3, imresize(imread(loc),inputSize),...
    imresize(imread(loc),inputSize), imresize(imread(loc),inputSize) );

% Training parameters
options = trainingOptions('sgdm',... 
    'MiniBatchSize',250,... 
    'MaxEpochs',30,... 
    'InitialLearnRate',0.00125,... 
    'LearnRateDropFactor',0.1,... 
    'LearnRateDropPeriod',20);

% Split the data into stratified subsets (split by label)
[subset{1} subset{2} subset{3} subset{4} subset{5} ...
    subset{6} subset{7} subset{8} subset{9} subset{10}] = ...
    splitEachLabel(imds, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);

% For each subset
for i=1:10
    fprintf('Subset %i:\n', i);
    % Randomly split subset into train and test data
    [train{i}, test{i}] = splitEachLabel(subset{i}, 0.7, 'randomized');
    % Train the network
    net = trainNetwork(train{i},layers,options);
    % Predict test values
    prediction{i} = classify(net, test{i});
end

% Save the results for analysis
save('alexnet.mat');
t2 = toc(t1);
fprintf('Success: %i minutes to complete.\n', floor(t2/60));
