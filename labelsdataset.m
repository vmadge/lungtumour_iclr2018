close all; clear; clc;

% t1 = tic;
% w = waitbar(0, 'Initializing...');

load('dataset/pixelmap.mat');
load('dataset/Img_loaded_not_labels.mat');

N = length(imds.Files);
labels = zeros(N,1);

% for i = 1:N
%     file = imds.Files{i};
%     SOP = file(end-67:end-4);
%     if isKey(pixels, SOP)
%         labels(i) = 1;
%     end
%     waitbar(i/N, w, sprintf('Building labels: %3.2f%%', 100*i/N));
% end
% 
% t2 = toc(t1);
% minutes = round(t2/60);
% close(w);
% fprintf('Success! %i minutes.\n', minutes);
% 
% save('dataset/labels.mat', 'labels');

A = pixels.keys;
for i = 1:length(A)
    A{i} = [A{i} '.png'];
end

[C1,ia1,ib1] = intersect(tmp(2,:),A);
labels(ia1) = 1;

save('dataset/labels.mat', 'labels');

imds.Labels = categorical(labels);
save('dataset/imagedataset.mat', 'imds');