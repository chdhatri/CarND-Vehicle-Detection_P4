# Vehicle Detection

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
Code for the project is in the Jupyter notebook project4.ipynb



### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Project conatins 2 datasets `vehicle` and `non-vehicle` images on which we will be extracting the Hog, color, spacial features to train the model in identifying the cars.  Sample dataset images containing car and non cars are below:

 ![png][./images/preview.png]

The code for the extratcing  features is contained in the 8th code cell of the IPython notebook (defined in 'extract_features' function) . Based on the flags spatial, color and HOG features were extracted. 

Code for extracting HOG features was defined in 5th code cell. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

 ![png][./images/hog.png]



#### 2. Explain how you settled on your final choice of HOG parameters.

Here is an example using the `YUV` color space and HOG parameters:
  * color_space = 'YUV'
  * spatial_size=(32, 32)
  * hist_bins=32
  * orient = 11
  * pix_per_cell = 16
  * cell_per_block = 2
  * hog_channel='ALL'


Final choice of HOG parameters were based on the performance of the SVM classifier predictions and the speed at which the classifier is able to make predictions.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
I trained the the linear SVM with the default classifier parameters and passing HOG, spatial and color channel features. I was able to acheive a test accuracy of 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search using the find_cars function in the notebook. This function was adapted from the lesson material.
This method instead of extracting HG features individually on each window it does extract on entire/part of the image. These full-image features are subsampled according to the size of the window and then fed to the classifier. 

This method performs the prediction based on the fed HOG features and returns list of rectangle windows drawn on to the positive predictions.

The image below shows the first attempt at using find_cars on one of the test images, using a single window size:

 ![png][./images/findcars.png]


I explored several options with multiple scales and window positions(y start and stop positions), with various overlaps in the X and Y directions.

The images below show the multiscale sliding window search by taking different scales, small (1x), medium (1.5x, 2x), and large (3x) windows: 
 ![png][./images/slidewindow1.png]
 ![png][./images/slidewindow2.png]

Below image with multiple bounding boxes reports positive detections. But we can notice that multiple overlapping detections exist for each of the two vehicles. 
 ![png][./images/combined_slidewindow.png]

To remove the duplicate detections and false positives , we will build a heat-map and thresholdfrom these detections in order to combine overlapping detections and remove false positives.The 'add_heat' function increments the pixel value (referred to as "heat") to (+=1) for all pixels within windows where a positive detection are reported by the classifier. The below image is the resulting heatmap from the detections in the image above:
 ![png][./images/heatmap.png]

A threshold is applied to the heatmap to reject the false positives. The result is below:
 ![png][./images/heatThresh.png]

To figure out how many cars are in each frame and which pixels belong to which cars,scipy.ndimage.measurements.label() function was called.
 ![png][./images/labels.png]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Some example images to demonstate pipeline is working:

 ![png][./images/process_framse.png]
---
The final implementation performs very well, identifying the near-field vehicles in each of the images with no false positives.

The original classifier used HOG features alone and achieved a test accuracy of 96.28%. I added spatial and color features to the original hog features and changed the channel to YUV channels whihc increased the accuracy to 98.40%,with a cost of increase in execution time. However, changing the pixels_per_cell parameter from 8 to 16 produced a roughly ten-fold increase in execution speed with minimal cost to accuracy.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for processing frames of video is contained in the cell titled "Pipeline for Processing Video Frames" and is identical to the code for processing a single image described above, with the exception of storing the detections (returned by find_cars) from the previous 15 frames of video using the prev_rects parameter from a class called Vehicle_Detect. Rather than performing the heatmap/threshold/label steps for the current frame's detections, the detections for the past 15 frames are combined and added to the heatmap and the threshold for the heatmap is set to 1 + len(det.prev_rects)//2 (one more than half the number of rectangle sets contained in the history) - this value was found to perform best empirically (rather than using a single scalar, or the full number of rectangle sets in the history).

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems that I faced while implementing this project were mainly concerned with detection accuracy. Balancing the accuracy of the classifier with execution speed was crucial. Scanning 190 windows using a classifier that achieves 98% accuracy should result in around 4 misidentified windows per frame. Of course, integrating detections from previous frames mitigates the effect of the misclassifications, but it also introduces another problem: vehicles that significantly change position from one frame to the next (e.g. oncoming traffic) will tend to escape being labeled. Producing a very high accuracy classifier and maximizing window overlap might improve the per-frame accuracy to the point that integrating detections from previous frames is unnecessary (and oncoming traffic is correctly labeled), but it would also be far from real-time without massive processing power.

The pipeline is probably most likely to fail in cases where vehicles (or the HOG features thereof) don't resemble those in the training dataset, but lighting and environmental conditions might also play a role (e.g. a white car against a white background). As stated above, oncoming cars are an issue, as well as distant cars (as mentioned earlier, smaller window scales tended to produce more false positives, but they also did not often correctly label the smaller, distant cars).

I believe that the best approach, given plenty of time to pursue it, would be to combine a very high accuracy classifier with high overlap in the search windows. The execution cost could be offset with more intelligent tracking strategies, such as:

determine vehicle location and speed to predict its location in subsequent frames
begin with expected vehicle locations and nearest (largest scale) search areas, and preclude overlap and redundant detections from smaller scale search areas to speed up execution
use a convolutional neural network, to preclude the sliding window search altogether
