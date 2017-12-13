close all; clear; clc;
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

save('datastore_unshuffled.mat', 'imds');

imds = shuffle(imds);

save('datastore.mat', 'imds');