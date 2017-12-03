%%% Costum CNN layers from paper

layers = [imageInputLayer([512 512 1],'Normalization','none','Name','inputl')
          convolution2dLayer([10 10],64,'NumChannels',1,'Stride',1,'Padding',5,'Name','conv1')
          reluLayer('Name','relu1')
          maxPooling2dLayer(3,'Name','max1') %%% ?????? Stride and Padding
          dropoutLayer(0.1,'Name','dropout1')
          convolution2dLayer([5 5],192,'NumChannels',1,'Stride',1,'Padding',5,'Name','conv2')
          reluLayer('Name','relu2')
          maxPooling2dLayer(3,'Name','max2') %%% ?????? Stride and Padding
          dropoutLayer(0.1,'Name','dropout2')
          convolution2dLayer([3 3],256,'NumChannels',1,'Stride',1,'Name','conv3')
          reluLayer('Name','relu3')
          convolution2dLayer([3 3],256,'NumChannels',1,'Stride',1,'Name','conv4')
          reluLayer('Name','relu4')
          convolution2dLayer([3 3],256,'NumChannels',1,'Stride',1,'Name','conv5')
          reluLayer('Name','relu5')
          convolution2dLayer([3 3],128,'NumChannels',1,'Stride',1,'Name','conv6')
          reluLayer('Name','relu6')
          maxPooling2dLayer(3,'Name','max3') %%% ?????? Stride and Padding
          dropoutLayer(0.5,'Name','dropout3')
          % fullyConnectedLayer(10,'Name','full2')
          softmaxLayer('Name','softm')
          classificationLayer('Name','out')];

%%
% load numerics dataset
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos',...
    'nndatasets','DigitDataset');
digitData = imageDatastore(digitDatasetPath,...
        'IncludeSubfolders',true,'LabelSource','foldernames');
%%
% split the data
minSetCount = min(digitData.countEachLabel{:,2})
trainingNumFiles = round(minSetCount/2);
rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(digitData,...
				trainingNumFiles,'randomize');      
%%

options = trainingOptions('sgdm','MaxEpochs',20,...
    'InitialLearnRate',0.001,'useGPU','no');

% just define a different set of layers for this particular dataset
layers = [imageInputLayer([28 28 1]);
          convolution2dLayer(5,20);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          fullyConnectedLayer(10);
          softmaxLayer();
          classificationLayer()];

%%

% run the convulation net
convnet = trainNetwork(trainDigitData,layers,options)

