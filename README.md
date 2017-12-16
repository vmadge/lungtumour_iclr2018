# openreview_lungtumouriclr2018
COMP 551 Final Project: Reproduce a paper from ICLR 2018.
Team JVZ

The following repository contains code used to reproduce results from the ICLR 2018 Conference Submission: LUNG TUMOR LOCATION AND IDENTIFICATION WITH ALEXNET AND A CUSTOM CNN

AlexNet.m is the 2017a MATLAB implementation of the famour CNN architecture. Results (alexnet.mat) were too large to upload to GitHub and thus can be found here: https://www.dropbox.com/s/7pmte04vw5b2gy8/alexnet.mat?dl=0

CNN.m, CNN3.m and CNN4.m are implementations of the CNN described by the paper, with different image resolutions. Ultimately, a resolution of 64x64 was used for the input layer, found in CNN5.m, which produced the results (cnn.mat) in the report.

BuildDatastore.m 

results.m compiles the performance metrics for the .mat files obtained after running the CNN and AlexNet. 

