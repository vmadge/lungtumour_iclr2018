# openreview_lungtumouriclr2018
COMP 551 Final Project: Reproduce a paper from ICLR 2018.
Team JVZ

The following repository contains code used to reproduce results from the ICLR 2018 Conference Submission: LUNG TUMOR LOCATION AND IDENTIFICATION WITH ALEXNET AND A CUSTOM CNN

1. dicom2png.m : Converts the original downloaded LIDC-IDRI dataset into a folder of PNG images, sorted by patient ID, and renamed using the SOP ID of each slice. 

2. convertLabels.m : Converts the xml labels within the dataset into a Matlab Map object, indexing each slice by it's SOP and including a matrix of each pixel labeled as a nodule. The script then creates a corresponding image mask for each slice, where 0 corresponds to benign tissue, and 1 corresponds to labeled nodules. This script utilizes the xml2struct implementation by Wouter Falkena (see included license file). Pixel label results are saved as pixelmap.mat.

3. buildDatastore.m : Builds a datastore object indexing all slices in the PNG folder, and assigning classification labels using the pixelmap.mat file produced by convertLabels.m. The datastore defines images by their absolute path, and thus cannot be copied from on machine to another. The datastore is saved as datastore.mat. 

4. AlexNet.m : The 2017a MATLAB implementation of the famous CNN architecture. Results (alexnet.mat) were too large to upload to GitHub and thus can be found here: https://www.dropbox.com/s/7pmte04vw5b2gy8/alexnet.mat?dl=0.

5. CNN5.m : CNN.m, CNN3.m and CNN4.m are implementations of the CNN described by the paper, with different image resolutions. Ultimately, a resolution of 64x64 was used for the input layer, found in CNN5.m, which produced the results (cnn.mat) in the report.

6. results.m : Compiles the performance metrics for the .mat files obtained after running the CNN and AlexNet. 
