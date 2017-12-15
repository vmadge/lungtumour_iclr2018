% Build a datastore.mat file on the current harddrive with the paths to
% each PNG image, and it's corresponding label using the pixelmap.mat

close all; clear; clc;

% Load the pixel Map from convertLabels.m
load('dataset/pixelmap.mat');
% Create a datastore from the PNG folder
imds = imageDatastore('dataset/PNG', 'IncludeSubfolders', true);
% Get the list of files
A = imds.Files;
N = length(A);
% Make a array for the labels
labels = zeros(N,1);

% Remove the file path and get only the SOP for the image
for i=1:N
    A{i} = A{i}(end-67:end-4);
end

% Find the indices (ia) of A that have nodule pixels in the pixel map
[C,ia,ib] = intersect(A,pixels.keys);
% Label all ia slices as cancerous
labels(ia) = 1;
imds.Labels = categorical(labels);

save('datastore_unshuffled.mat', 'imds');

% Randomize the dataset order
imds = shuffle(imds);

% Save the datastore to be accessed by AlexNet/CNN
save('datastore.mat', 'imds');