% Compute accuracy metrics from the saved results of AlexNet or CNN

close all; clear; clc;
load('alexnet.mat');
% load('cnn.mat');

% Build a composite array of all predictions/labels
ot = [];
yt = [];

% For each subset
for i = 1:10
    % Index Subset
    pred = prediction{i};
    label = test{i}.Labels;
    
    % Categorical variables into integers
    o = grp2idx(pred) - 1; % Prediction
    y = grp2idx(label) - 1; % Label
    
    % Append total array
    ot = [ot;o];
    yt = [yt;y];
    
    T = o == y; % True/correct indices
    n = length(T); 
    
    TP = sum(T & o); % True Positives
    TN = sum(T & ~o); % True Negatives
    FP = sum(~T & o); % False Positives
    FN = sum(~T & ~o); % False negatives
    
    % Accuracy metrics
    accuracy = (TP + TN)/(TP + TN + FP + FN);
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1 = 2* (recall*precision)/(recall+precision);
    MCC = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
    FPR = FP/(TN+FP);
    FNR = FN/(TP+FN);
    
    % Print results
    text = ['Subset %i: \n', ...
        'Accuracy: %3.2f%% \n', ...
        'Precision: %3.2f%% \n', ...
        'Recall: %3.2f%% \n', ...
        'F1: %3.2f \n', ...
        'MCC: %3.2f \n', ...
        'FPR: %3.2f \n', ...
        'FNR: %3.2f \n', ...
        '\n\n'];
    
    fprintf(text, i, accuracy*100, precision*100, recall*100, ...
        F1, MCC, FPR, FNR);
    
end

% Repeat for the cumulative array to get the total scores
T = ot == yt;
n = length(T);

TP = sum(T & ot);
TN = sum(T & ~ot);
FP = sum(~T & ot);
FN = sum(~T & ~ot);

accuracy = (TP + TN)/(TP + TN + FP + FN);
precision = TP/(TP+FP);
recall = TP/(TP+FN);
F1 = 2* (recall*precision)/(recall+precision);
MCC = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
FPR = FP/(TN+FP);
FNR = FN/(TP+FN);

text = ['Total: \n', ...
    'Accuracy: %3.2f%% \n', ...
    'Precision: %3.2f%% \n', ...
    'Recall: %3.2f%% \n', ...
    'F1: %3.2f \n', ...
    'MCC: %3.2f \n', ...
    'FPR: %3.2f \n', ...
    'FNR: %3.2f \n', ...
    '\n\n'];

fprintf(text, accuracy*100, precision*100, recall*100, ...
    F1, MCC, FPR, FNR);