close all; clear; clc;
load('cnn.mat');

ot = [];
yt = [];

for i = 1:10
    pred = prediction{i};
    label = test{i}.Labels;
    
    o = grp2idx(pred) - 1;
    y = grp2idx(label) - 1;
    
    ot = [ot;o];
    yt = [yt;y];
    
    T = o == y;
    n = length(T);
    
    TP = sum(T & o);
    TN = sum(T & ~o);
    FP = sum(~T & o);
    FN = sum(~T & ~o);
    
    accuracy = (TP + TN)/(TP + TN + FP + FN);
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    F1 = 2* (recall*precision)/(recall+precision);
    MCC = (TP*TN - FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
    FPR = FP/(TN+FP);
    FNR = FN/(TP+FN);
    
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