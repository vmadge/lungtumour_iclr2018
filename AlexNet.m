close all; clear; clc;

t1 = tic;

load('dataset/pixelmap.mat');

imds = imageDatastore('dataset/PNG', 'IncludeSubfolders', true);
A = imds.Files;
N = length(A);
labels = zeros(N,1);

for i=1:N
    A{i} = A{i}(end-67:end-4);
end

[C,ia,ib] = intersect(A,pixels.keys);
labels(ia) = 1;
imds.Labels = categorical(labels);


net = alexnet;

layers = net.Layers(1:end-3);
layers(end+1) = fullyConnectedLayer(2,... 
    'WeightLearnRateFactor',10,... 
    'BiasLearnRateFactor',20); 
layers(end+1) = softmaxLayer(); 
layers(end+1) = classificationLayer();

inputSize = net.Layers(1).InputSize(1:2);
imds.ReadFcn = @(loc)cat(3, imresize(imread(loc),inputSize),...
    imresize(imread(loc),inputSize), imresize(imread(loc),inputSize) );

options = trainingOptions('sgdm',... 
    'MiniBatchSize',250,... 
    'MaxEpochs',30,... 
    'InitialLearnRate',0.00125,... 
    'LearnRateDropFactor',0.1,... 
    'LearnRateDropPeriod',20);


n = 10;
for i=1:n
    fprintf('Subset %i:\n', i);
    subset{i} = partition(imds, n, i);
    [train{i}, test{i}] = splitEachLabel(subset{i}, 0.7, 'randomized');
    net = trainNetwork(train{i},layers,options);
    prediction{i} = classify(net, test{i});
end

save('alexnet.mat');
t2 = toc(t1);
fprintf('Success: %i minutes to complete.\n', floor(t2/60));
