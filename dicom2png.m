close all; clear; clc;

%Initialize waitbar
w = waitbar(0, 'Initializing...');

% Clear existing PNG data/directory
if exist('dataset/PNG', 'dir')
    rmdir('dataset/PNG', 's');
end

% Initialize PNG directory
mkdir('dataset/PNG');

% Get list of each series as a subdirectory of LIDC-IDRI
series = dir('dataset/LIDC-IDRI');
series = series(4:length(series)); %length(series) % First 3 folders are '.,..,.DS_Store'

% For each series/subfolder in the dataset
for i=1:length(series)
    
    % Make a matching PNG subfolder
    mkdir(['dataset/PNG/' series(i).name]);
    
    % Define folder path
    folder = [series(i).folder '/' series(i).name];
    
    % Get the label file
    labels = dir([folder '/**/*.xml']);
    % Get list of each dicom image in the CT scan
    dicoms = dir([folder '/**/*.dcm']);
    
    % In the event of multiple subfolders/label files per series
    for j=1:length(labels)
        % Index the label file
        label = labels(j);
        
        % Read the label xml into matlab
        xDoc = xmlread([label.folder '/' label.name]);

        %Convert xml labels --TODO

        % Output the updated xml file
        xml_outfile = ['dataset/PNG/' series(i).name '/' label.name];
        xmlwrite(xml_outfile, xDoc);
    end
    
    % For each dicom image in the CT
    for j=1:length(dicoms)
        
        % Define path to dicom
        dicom = [dicoms(j).folder '/' dicoms(j).name];
        
        % Read dicom header/metadata
        header = dicominfo(dicom);
        % Read dicom image
        I = dicomread(dicom);
        
        % Scale image to grayscale [0,1], then to uint8 [0,255]
        Ib8 = uint8(255 * mat2gray(I));
        
        % The xml file references images by their SOP UID
        % Get the SOP for the image
        SOP = header.SOPInstanceUID;
        
        % Define the output directory and filename using the SOP UID
        png = ['dataset/PNG/' series(i).name '/' SOP '.png']; %dicoms(j).name(1:end-4)
        % Write the image as a .png file
        imwrite(Ib8, png, 'png');
        
        % Update waitbar
        step = j + (i-1)*length(dicoms);
        steps = length(series)*length(dicoms);
        waitbar(step/steps, w, ...
            sprintf('Converting: %3.2f%%', 100*step/steps));
        
        % Pixel Spacing in each dimension
        psx = header.PixelSpacing(1);
        psy = header.PixelSpacing(2);
        
    end
    
end

close(w);
fprintf('Success!\n');

