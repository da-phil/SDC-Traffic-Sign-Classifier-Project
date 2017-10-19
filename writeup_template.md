# Traffic Sign Recognition

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[dataset-example]: ./examples/dataset_example.png "Visualization of dataset"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/dataset_distribution_chart.png "Dataset sample distribution"

### Data Set Summary & Exploration

**Stats of the dataset**

* Image data shape:  32x32 RGB pixel
* Number of classes: 43

|   Dataset           |   # Samples  |   Percentage of whole dataset |
|--------------------:|-------------:|-------------------------------:|
|     Training dataset |  34799  | 67.13% |
|   Validation dataset |  4410  | 8.51% |
|         Test dataset |  12630  | 24.36% |
|              **Total** |  51839                                ||

![Dataset sample distribution][image5]


**Number of unique classes/labels in the training, validation and test data sets**

|    | Traffic sign description                           |   Training |   Validation |   Test |
|---:|:---------------------------------------------------|-----------:|-------------:|-------:|
|  0 | Speed limit (20km/h)                               |        180 |           30 |     60 |
|  1 | Speed limit (30km/h)                               |       1980 |          240 |    720 |
|  2 | Speed limit (50km/h)                               |       2010 |          240 |    750 |
|  3 | Speed limit (60km/h)                               |       1260 |          150 |    450 |
|  4 | Speed limit (70km/h)                               |       1770 |          210 |    660 |
|  5 | Speed limit (80km/h)                               |       1650 |          210 |    630 |
|  6 | End of speed limit (80km/h)                        |        360 |           60 |    150 |
|  7 | Speed limit (100km/h)                              |       1290 |          150 |    450 |
|  8 | Speed limit (120km/h)                              |       1260 |          150 |    450 |
|  9 | No passing                                         |       1320 |          150 |    480 |
| 10 | No passing for vehicles over 3.5 metric tons       |       1800 |          210 |    660 |
| 11 | Right-of-way at the next intersection              |       1170 |          150 |    420 |
| 12 | Priority road                                      |       1890 |          210 |    690 |
| 13 | Yield                                              |       1920 |          240 |    720 |
| 14 | Stop                                               |        690 |           90 |    270 |
| 15 | No vehicles                                        |        540 |           90 |    210 |
| 16 | Vehicles over 3.5 metric tons prohibited           |        360 |           60 |    150 |
| 17 | No entry                                           |        990 |          120 |    360 |
| 18 | General caution                                    |       1080 |          120 |    390 |
| 19 | Dangerous curve to the left                        |        180 |           30 |     60 |
| 20 | Dangerous curve to the right                       |        300 |           60 |     90 |
| 21 | Double curve                                       |        270 |           60 |     90 |
| 22 | Bumpy road                                         |        330 |           60 |    120 |
| 23 | Slippery road                                      |        450 |           60 |    150 |
| 24 | Road narrows on the right                          |        240 |           30 |     90 |
| 25 | Road work                                          |       1350 |          150 |    480 |
| 26 | Traffic signals                                    |        540 |           60 |    180 |
| 27 | Pedestrians                                        |        210 |           30 |     60 |
| 28 | Children crossing                                  |        480 |           60 |    150 |
| 29 | Bicycles crossing                                  |        240 |           30 |     90 |
| 30 | Beware of ice/snow                                 |        390 |           60 |    150 |
| 31 | Wild animals crossing                              |        690 |           90 |    270 |
| 32 | End of all speed and passing limits                |        210 |           30 |     60 |
| 33 | Turn right ahead                                   |        599 |           90 |    210 |
| 34 | Turn left ahead                                    |        360 |           60 |    120 |
| 35 | Ahead only                                         |       1080 |          120 |    390 |
| 36 | Go straight or right                               |        330 |           60 |    120 |
| 37 | Go straight or left                                |        180 |           30 |     60 |
| 38 | Keep right                                         |       1860 |          210 |    690 |
| 39 | Keep left                                          |        270 |           30 |     90 |
| 40 | Roundabout mandatory                               |        300 |           60 |     90 |
| 41 | End of no passing                                  |        210 |           30 |     60 |
| 42 | End of no passing by vehicles over 3.5 metric tons |        210 |           30 |     90 |

This table exposes a drastic inbalance between the sample-sizes across all classes. I'd expect that training on this dataset might lead to biased classification and weak recognition of classes with less examples.

#### Exploratory visualization of the dataset

![Dataset visualization][dataset-example]


### Design and Test a Model Architecture

#### Preprocessing pipeline

**Colorspaces considerations**
As the traffic signs are available as RGB images it makes sense to stick to RGB colorspace and use it for training.
Unlike other colorspaces RGB should give the trained network the same intuition as the human when it comes to color perception,
e.g. it make take advantage of the color-coding of particular signs as additional cues for the classification, besids the structure and patterns.

On the other hand using colorspaces which seperate the luminance (or brightness) from the chrominance (or colors) allow for tweaking the image contrast while not changing the colors. Therefore the YUV and LAB colorspaces were considered, where the luminance channel was used for adaptive histogram equalization using `cv2.createCLAHE` function, the color channels were left as is. My assumption was that the network would benefit from a better image contrast, extracting patterns and structure more easily.

**Image equalization and normalization**
Images 

**Augmented dataset example**

The original dataset was augmented using `ImageDataGenerator` from `keras.preprocessing.image` in order to make the trained classifier more robost against deviations from the training dataset.
The following augmentations were applied:
* Rotations: [0, 20]°
* Image shifts in x and y axis: [0, 0.1] %
* Zoom: [0, 0.1] %

IMAGE orig -> IMAGE in new colorspace -> IMAGE augmented



#### Model selection

The first choice of course was the recommended LeCun5 network from the course material.
It delivered a validation accuracy of around 90% while using unmodified training data but didn't really generalize well on new images.
Besides it was impossible to get a validation accuracy higher than approx. 80% on the augmented dataset.

In order to enable the network to learn more features and generalize better I added one more convolution + max pooling layer and increased the depth of the convolution layers.

My final model consisted of the following layers:

| Layer         		      |     Description                         | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x3 RGB image                       |
| Convolution           | kernel: 5x5, stride: 1x1, output: 32x32x64    |
| Max pooling           | kernel: 2x2, stride: 2x2, output: 16x16x64    |
| Convolution           | kernel: 5x5, stride: 1x1, output: 16x16x64    |
| Max pooling           | kernel: 2x2, stride: 2x2, output: 8x8x256     |
| Convolution           | kernel: 5x5, stride: 1x1, output: 8x8x256     |
| Max pooling           | kernel: 2x2, stride: 2x2, output: 4x4x256     |
| Fully connected       | input: 4096, output: 400                      |
| RELU                  |                                               |
| Dropout               | keep_prob: 0.6                                |
| Fully connected       | input: 400, output: 200                       |
| RELU                  |                                               |
| Dropout               | keep_prob: 0.6                                |
| Fully connected out   | input: 200, output: 43                        |    
    
#### Training

optimizer:
the batch size: 
number of epochs:
learning rate:


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


