# **Behavioral Cloning Project**

The goals of this project are the following:
* Use a driving simulator to collect data for good driving behavior. The data collect includes reinforcing data for different scenarios like curves, recovering to central driving, etc.
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/nvidia_model.png "CNN Architecture"
[image2]: ./images/normal_driving.jpg "Normal Driving"
[image3]: ./images/recovery_1.jpg "Recovery Image"
[image4]: ./images/recovery_2.jpg  "Recovery Image"
[image5]: ./images/recovery_3.jpg  "Recovery Image"
[image6]: ./images/flip_1.png "Normal Image"
[image7]: ./images/flip_2.png "Flipped Image"


## Files & Code

### 1. The project includes the following files:
* model.py - contains the code for training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.
* drive.py - contains the code for driving the car in autonomous mode
* model.h5 - contains trained convolution neural network 


### 2. Submission includes functional code
Using a [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip) and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

## Model Architecture and Training Strategy

### 1. Model architecture

The model used is the [NVIDIA published Architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and consists of a set of 9 layers:
* A normalization layer using a Keras lambda layer (model.py lines 104) 
* 5 convolutional layers using a RELU activation function (model.py lines 106-110) 
* 2 dropout layers (rate =50%) for the last 2 convolutional layers
* 3 fully connected layers (model.py lines 112-114) 

The NVIDIA CNN architecture was selected as baseline model. After some iterations, though, it was decided to use the NVIDIA model 'as is', since good driving performance was achieved for this project.

![alt text][image1]


|Layer (type)|Output Shape|Param # |
|:---------------------:|:---------------------------------------------:| 
|lambda_1 (Lambda)      |(None, 160, 320, 3)     |  0         |
|cropping2d_1 (Cropping2D)  |  (None, 80, 320, 3)      |  0         |
|conv2d_1 (Conv2D)          |  (None, 38, 158, 24)     |  1824      |
|conv2d_2 (Conv2D)          |  (None, 17, 77, 36)      |  21636     |
|conv2d_3 (Conv2D)          |  (None, 7, 37, 48)       |  43248     |
|conv2d_4 (Conv2D)          |  (None, 5, 35, 64)       |  27712     |
|dropout_1 (Dropout)        | (None, 5, 35, 64)        |    0 | 
|conv2d_10 (Conv2D)         |  (None, 3, 33, 64)       |  36928     |
|dropout_1 (Dropout)        | (None, 3, 33, 64)        |    0 | 
|flatten_2 (Flatten)        |  (None, 6336)            |  0         |
|dense_5 (Dense)            |  (None, 100)             |  633700    |
|dense_6 (Dense)            |  (None, 50)              |  5050      |
|dense_7 (Dense)            |  (None, 10)              |  510       |
|dense_8 (Dense)            |  (None, 1)               |  11        |

*Total params: 770,619.0*

*Trainable params: 770,619.0*

*Non-trainable params: 0.0*


### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 78-80). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Two dropout layers (50%) were introduced for the last 2 convolutional layers to reduce overfitting.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 124).

### 4. Training data

The key strategy on deriving a good driving behavior was focused on gathering quality data. 

The following steps were followed in order to collect driving data to train the model. Note that this was a trial-and-error process. A few data points were captured and fed into the model and tested on the simulator. Base on observances, more data was captured.

1) Center lane driving - 3 laps were recorded using center lane driving. Here is an example image of center lane driving:
![alt text][image2]

2) Counterclockwise driving - 1 lap was recorded using center lane driving counterclockwise - Goal was to generalize.

3) Recovering from the left and right sides of the road - 1 lap was recorded recovering the car from the side to the center of the lane. Note that only car recovering from the sides was recorded, and it was randomly recorded from left to center and right to center. These images show what a recovery looks like starting from teh right:
![alt text][image3]
![alt text][image4]
![alt text][image5]

4) Driving through curves - 1 lap was recorded driving only through curves using slower speeds.

5) Reinforcing driving through 'trouble' area - A few pases were recorded around trouble areas like sharp curves.

After the collection process, the number of data points was 13650 . 
A function was created to augment images by fliping all images and angles. (model.py line 39-46). The total number of images after augmentation was 27300.

Here is a sample of a 'normal' and 'flipped' image:
![alt text][image6]
![alt text][image7]

The data was randomly shuffled the data to put 20% into a validation set. 

The training data was used for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by ploting the loss vs epocs curves for the training and validation sets.

* After training the model with data from 1) to 2), it was observed that the car would perform OK on straight sections, but would not perform OK on curves, and will not return to the center of the lane after moving to the sides.

* After training the model with data from 1) thorugh 4), it was observed that the car would perform OK through most of the road, but there were some problematic areas such as sharp curves (either not steering enough or getting into the curve too close to one side making steering hard).

* After training the model with data from 1) through 5), it was observes that the car would drive around the road. 
