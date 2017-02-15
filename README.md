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

![Example training images](./images/training_data_vehicle.png)

When loading the images using matplotlib.image the image will automatically be normalized. I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Below an image from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![vehicle example](./images/hog_features_vehicle.png)

![no vehicle example](./images/hog_features_no_vehicle.png)

The final paramters are based on emperical results. I've tried multiple combinations, see a subset of the results in the table below. The colour channels tested include RGB, HSL, HSV, YUV, and YCrCB.

The best result was obtained using .. 9 orientations , 8 pixel per cell and 2 cell per blocks

The code for reading the image and extracting the HOG features can be found in the functions read_image() and extract_hog_features() in the file 'process_image.py'. These steps are made visually in the Notebook Process Image.

I trained a SVM with linear kernel using the SVC implementation in scikit learn (with all default paramters). This model showed a test accuracy of 0.991.

## Sliding-window technique
###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To detect car in our image, I've used a sliding window technique. The code can be found in the function ...

My final model uses a window size of .x. and the model only searches the bottom of the image. We only use x-values between . and ., because the top only represents the sky and some background and the bottom the front of the car. 
The final paramters are chosen based on emperical results with brute force techniques. Values between . and . are tried for the window size. For the overlap we decided to go for . for the same reason (tried between . x .). The higher the overlap the more compute expensive the algorithms is. For example, x windows for x and x windows for x.

![alt text][image3]

These images are resized before feeding into the classifier.



####Some discussion is given around how you improved the reliability of the classifier i.e., fewer false positives and more reliable car detections (this could be things like choice of feature vector, thresholding the decision function, hard negative mining etc.)

![Heatmap and vehicle detection example 1](/images/heatmap_detection_example1.png)
![Heatmap and vehicle detection example 2](/images/heatmap_detection_example2.png)
![Heatmap and vehicle detection example 3](/images/heatmap_detection_example3.png)




## Video pipeline
####A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. Same or similar method used to draw bounding boxes (or circles, cubes, etc.) around high-confidence detections where multiple overlapping detections occur.


To make my algorithm more robust for videos I've added some techniques to detect vehicles in subsequent frames. A heatmap is added to show the location of repeated vehicle detecteions. This is used to reduce the number of false positives. 


## Output video


The output video can be downloaded from [here](output_video.mp4).

## Discussion
To improve the classifier:
* Try other algorithms
* Fine tune parameters
