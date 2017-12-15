% Create a pixel Map object of all the pixel labels in the dataset, and
% create a mask image for each PNG in the dataset

close all; clear; clc;
t1 = tic;

%----- Generate a Pixel Map from Labels -----%

%Initialize waitbar
w = waitbar(0, 'Initializing...');
 
% Get all label files
labels = dir('dataset/PNG/**/*.xml');

% Pixel map to store ROIs for each image
pixels = containers.Map();

for i=1:length(labels)
    % Index the label file
    label = labels(i);

    % Read the label xml into matlab
    root = xml2struct([label.folder '/' label.name]);
    
    % Check that there are labels to read
    if ~isfield(root.LidcReadMessage, 'readingSession')
        continue;
    end
    sessions = root.LidcReadMessage.readingSession;
    
    % For each expert reader
    for j=1:length(sessions)
        session = sessions{j};
        
        % Check if there are nodules
        if ~isfield(session, 'unblindedReadNodule')
            continue;
        end
        nodules = session.unblindedReadNodule;
        
        % For each nodule identified
        for k=1:length(nodules)
            if length(nodules) == 1
                nodule = nodules;
            else
                nodule = nodules{k};
            end
            rois = nodule.roi;
            
            % For each region of interest
            for l=1:length(rois)
                if length(rois) == 1
                    roi = rois;
                else
                    roi = rois{l};
                end
                
                % Only proceed if the ROI is marked as included
                if roi.inclusion.Text ~= string('TRUE')
                    continue;
                end
                
                % Get the SOP
                SOP = roi.imageSOP_UID.Text;
                
                % Get the edgeMap pixels
                edgeMaps = roi.edgeMap;
                xy = zeros(length(edgeMaps),2);
                for m=1:length(edgeMaps)
                    if length(edgeMaps) == 1
                        edgeMap = edgeMaps;
                    else
                        edgeMap = edgeMaps{m};
                    end
                    % Get edgemap coordinates
                    x = str2double(edgeMap.xCoord.Text);
                    y = str2double(edgeMap.yCoord.Text);
                    % Add coordinates to ROI matrix
                    xy(m,:) = [x,y]; 
                end
                
                % Append ROI matrix to any preexisting coordinates
                if isKey(pixels,SOP)
                    xy = [pixels(SOP);xy];
                end
                
                % Store ROIs in pixel map
                pixels(SOP) = xy;
                
            end
        end
    end
    
    % Update waitbar
    step = i;
    steps = length(labels);
    waitbar(step/steps, w, ...
        sprintf('Reading labels: %3.2f%%', 100*step/steps));
    
end

save('dataset/pixelmap.mat', 'pixels');

close(w);
fprintf('Success!\n');

%----- Generate Images Masks as Labels from Pixel Map -----%

% load('pixelmap.mat');

%Initialize waitbar
w = waitbar(0, 'Initializing...');

% Clear existing labels data/directory
if exist('dataset/labels', 'dir')
    rmdir('dataset/labels', 's');
end

% Initialize labels directory
mkdir('dataset/labels');

% Get list of each series as a subdirectory of LIDC-IDRI
series = dir('dataset/PNG');
series = series(4:length(series)); %length(series) % First 3 folders are '.,..,.DS_Store'

% For each series/subfolder in the dataset
for i=1:length(series)
    
    % Make a matching labels subfolder
    mkdir(['dataset/labels/' series(i).name]);
    
    % Define folder path
    folder = [series(i).folder '/' series(i).name];
    
    % Get list of each dicom image in the CT scan
    pngs = dir([folder '/*.png']);
    
    % For each png image in the CT
    for j=1:length(pngs)
        
        % Define path to png
        png = [pngs(j).folder '/' pngs(j).name];
        
        % Define SOP
        SOP = pngs(j).name(1:end-4);
        
        % Read image
        I = imread(png);
        % Initialize mask image
        L = zeros(size(I));
        
        % Read in pixels for labeled images
        if isKey(pixels,SOP)
            xy = pixels(SOP);
            for k = 1:size(xy,1)
                L(xy(k,2),xy(k,1)) = 1;
            end
            L = imfill(L);
        end
        
        % Define the output directory and filename using the SOP UID
        label = ['dataset/labels/' series(i).name '/' SOP '.png']; %dicoms(j).name(1:end-4)
        % Write the image as a .png file
        imwrite(L, label, 'png');
        
        % Update waitbar
        step = j + (i-1)*length(pngs);
        steps = length(series)*length(pngs);
        waitbar(step/steps, w, ...
            sprintf('Creating masks: %3.2f%%', 100*step/steps));

    end
    
end

t2 = toc(t1);
minutes = round(t2/60);

close(w);
fprintf('Success! %i minutes.\n', minutes);
