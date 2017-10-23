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
[dataset-distribution]: ./examples/dataset_distribution_chart.png "Dataset sample distribution"
[classes-distribution]: ./examples/dataset_class_distribution_chart.png "Distribution of classes in datasets"
[image-augmentations]: ./examples/image_augmentations.png "Example showing image augmentations, all randomly applied"
[equalization1]: ./examples/equalization1.png "Equalization with fixed clipLimit and variable grid_size"
[equalization2]: ./examples/equalization2.png "Equalization with variable clipLimit and fixed grid_size"
[featuremap_conv1_1]: ./examples/featuremap_conv1_1.png
[featuremap_conv2_1]: ./examples/featuremap_conv2_1.png

[sign1]: ./extra-examples/sign1.jpg
[sign2]: ./extra-examples/sign2.jpg
[sign3]: ./extra-examples/sign3.jpg
[sign4]: ./extra-examples/sign4.jpg
[sign5]: ./extra-examples/sign5.jpg
[sign6]: ./extra-examples/sign6.jpg
[sign7]: ./extra-examples/sign7.jpg
[sign8]: ./extra-examples/sign8.jpg

[pred_tf1]: ./examples/pred_tf0.png
[pred_tf2]: ./examples/pred_tf1.png
[pred_tf3]: ./examples/pred_tf2.png
[pred_tf4]: ./examples/pred_tf3.png
[pred_tf5]: ./examples/pred_tf4.png
[pred_tf6]: ./examples/pred_tf5.png
[pred_tf7]: ./examples/pred_tf6.png
[pred_tf8]: ./examples/pred_tf7.png
[pred_tf_test1]: examples/pred_tf_test1.png
### Data Set Summary & Exploration

**Stats of the dataset**

* Image data shape:  32x32 RGB pixel
* Number of classes: 43

![dataset-distribution]

|   Dataset           |   # Samples  |   Percentage of whole dataset |
|--------------------:|-------------:|-------------------------------:|
|     Training dataset |  34799  | 67.13% |
|   Validation dataset |  4410  | 8.51% |
|         Test dataset |  12630  | 24.36% |
|              **Total** |  51839                                ||


**Number of unique classes/labels in the training, validation and test data sets**

![classes-distribution]

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
For example the "Speed limit (20km/h)" as well as the "Go straight or left" sign is underrepresented in all three datasets.

In order to overcome a biased prediction I balanced the training dateset by equalizing the number of training examples per class, i.e. adding copies of instances from the under-represented class, called over-sampling, or more formally sampling with replacement.
As a reference count I used the count of the class which has the most samples (class 2, samples = 2010).
Having 2010 samples from all 43 classes leads to a new balanced training set of 86430 samples. In the random image autmentation later I make sure that all the samples which appear several times in the training set look different due to the random augmentations.

The following articles discusses other solutions to an inbalanced training dataset, for example it penalizes the reward for the majority class, that means re-weighting the loss function (3rd link):
* https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html
* https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset
* https://blog.fineighbor.com/tensorflow-dealing-with-imbalanced-data-eb0108b1070 (loss re-weighting example)

#### Exploratory visualization of the dataset

![Dataset visualization][dataset-example]


### Design and Test a Model Architecture

#### Preprocessing pipeline

**Colorspaces considerations**
As the traffic signs are available as RGB images it makes sense to stick to RGB colorspace and use it for training.
Unlike other colorspaces RGB should give the trained network the same intuition as the human when it comes to color perception,
e.g. it make take advantage of the color-coding of particular signs as additional cues for the classification, besids the structure and patterns.

On the other hand using colorspaces which seperate the luminance (or brightness) from the chrominance (or colors) allow for tweaking the image contrast while not changing the colors. Therefore the YUV and LAB colorspaces were considered, where the luminance channel was used for adaptive histogram equalization using `cv2.createCLAHE` function, the color channels were left as is. My assumption was that the network would benefit from a better image contrast, extracting patterns and structure more easily.

**Image histogram equalization and normalization**
Using the function `cv2.createCLAHE(clipLimit, tileGridSize)` I equalized the histograms of the luminance channels of the YUV and LAB colorspaces and the B/W images.
I tried to find suitable parameters for `clipLimit`, `tileGridSize` by plotting images where I keep `clipLimit` fixed and `tileGridSize` variable and vice versa.

Plot with `clipLimit=2` and variable `tileGridSize`:

![equalization1]

Plot with `tileGridSize=(4,4)` and variable `clipLimit`

![equalization2]

In the end I chose `clipLimit=5` and `tileGridSize=(5,5)`.

Normalization was done as suggested, by this function: `(pixel - 128) / 128`

**Augmented dataset example**

The original dataset was augmented using `ImageDataGenerator` from `keras.preprocessing.image` in order to make the trained classifier more robost against deviations from the training dataset.
The following augmentations were applied:
* Rotations: [0, 20]°
* Image shifts in x and y axis: [0, 0.1] %
* Zoom: [0, 0.1] %

![image-augmentations]



#### Model selection

The first choice of course was the recommended LeCun5 network from the course material.
It delivered a validation and test accuracy of slightly above 90% while using unmodified training data but didn't really generalize well on new images (accuracy was approx. 50%).

In order to enable the network to learn more features and generalize better I added one more convolution + max pooling layer and increased the depth of the convolution layers.

My final model consisted of the following layers:

| Layer         		      |     Description                         | 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x3 RGB image                       |
| Convolution           | kernel: 5x5, stride: 1x1, output: 32x32x32    |
| Max pooling           | kernel: 2x2, stride: 2x2, output: 16x16x32    |
| Convolution           | kernel: 5x5, stride: 1x1, output: 16x16x64    |
| Max pooling           | kernel: 2x2, stride: 2x2, output: 8x8x64     |
| Convolution           | kernel: 5x5, stride: 1x1, output: 8x8x128     |
| Max pooling           | kernel: 2x2, stride: 2x2, output: 4x4x128     |
| Fully connected       | input: 2048, output: 400                      |
| RELU                  |                                               |
| Dropout               | keep_prob: 0.6                                |
| Fully connected       | input: 400, output: 200                       |
| RELU                  |                                               |
| Dropout               | keep_prob: 0.6                                |
| Fully connected out   | input: 200, output: 43                        |    
    
#### Training

* Optimizer: Adam
* Batch size: 248
* Number of epochs: 60
* Learning rate: 0.001

Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

In the beginning I was training the network on the normal training dataset and 

My final model results were:
* training set accuracy: 0.998
* validation set accuracy: 0.981
* test set accuracy: 0.972

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
 
 Testing on the test dataset gives an accuracy of 97.4% accuracy, where the accuracies are distributed among the class as follows:

![pred_tf_test1]

This distribution clearly shows that the network might have the largest errors with classes 27 and 30 when evaluated on new images.

### Test a Model on New Images

#### Picking random German traffic signs from the web
I chose 8 random German traffic signs which I found on the web and also included one for testing one of the "weakly" trained classes (class 27 - Pedestrians) to test the hypothesis from above.

![sign1] ![sign2] ![sign3] ![sign4] ![sign5] ![sign6] ![sign7] ![sign8]

The second and forth image might be difficult to classify correctly, because the second sign was slightly modified (head of person) and the forth is strongly rotated.

#### Discussion of prediction results
Using the random traffic signs above and feeding them into the trained network gave the following softmax probabilities as predictions:

![pred_tf1]

![pred_tf2]

![pred_tf3]

![pred_tf4]

![pred_tf5]

![pred_tf6]

![pred_tf7]

![pred_tf8]


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%, which is an improvement over the 50% accuracy with the original LeCun5 network.
However even after several attempts to re-train the network it always got the "Pedestrian" sign (class 27) wrong, even though I'm training the network with a balanced dataset. Maybe other techniques mentioned in the links further up might be worth looking at.


### Visualization of the networks featuremaps

Visualization of the first and second convolutional layers after evaluating the model only on "Speed limit (20km/h)" (class 0) signs.

#### ConvLayer1:

![featuremap_conv1_1]

#### ConvLayer2:

![featuremap_conv2_1]
