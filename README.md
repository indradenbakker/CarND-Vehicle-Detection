# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project I've described the steps to detect vehicles in a video. The steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Histogram of Oriented Gradients (HOG) feature extraction
The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images that were shared as part of this project. Below an example of one of each of the `vehicle` and `non-vehicle` classes:

![Example training images][/images/..]

When loading the images using matplotlib.image the image will automatically be normalized. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Beleow an image from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

The final paramters are based on emperical results. I've tried multiple combinations, see a subset of the results in the table below. The colour channels tested include RGB, HSL, HSV, YUV, and YCrCB.

The best result was obtained using .. 9 orientations , 8 pixel per cell and 2 cell per blocks

The code for reading the image and extracting the HOG features can be found in the functions read_image() and extract_hog_features() in the file 'process_image.py'. These steps are made visually in the Notebook Process Image.

I trained a SVM with linear kernel using the SVC implementation in scikit learn (with all default paramters). This model showed a test accuracy of 0.991.

## Sliding-window technique
###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To detect car in our image, I've used a sliding window technique.

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The HOG feature extractor is computational expensive, so I've decided to limit the search space. For example, the top .. pixels of the image useally don't contain any cars (mostly sky and background). Also, the bottom x pixels show the front of the car so we can exclude that part as well. 

For my sliding window I've used a window size of x with an overlap of 80%. 

These images are resized and passed to the HOG features extractor before feeding into the classifier.


Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

####Some discussion is given around how you improved the reliability of the classifier i.e., fewer false positives and more reliable car detections (this could be things like choice of feature vector, thresholding the decision function, hard negative mining etc.)

## Video pipeline



## Output video



## Discussion

