# Temperature_Forecast_Super_Resolution

Increasing Fine-Scale Temperature Details from Weather Model Forecasts

# PROBLEM STATEMENT

This challenge is to increase the resolution (the level of detail)of 2D surface temperature forecasts obtained from Environment and Climate Change Canada (ECCC) ’s weather forecast model, using as labelled images 2D temperature analysis at higher resolution. The scale factor between the model and the higher resolution analysis is 4 (from 10 km to 2.5 km). Numerous and relevant 2D low-resolution weather forecast fields are provided as predictors in the training set. In addition to temperature, these include fields like cloud coverage, wind, humidity, topography, etc. are also included in the training set files.

 ![alt text](https://github.com/Nishant-Chhetri/Temperature_Forecast_Super_Resolution/blob/master/content_MSC_image.jpg)

Here are examples of a temperature field over Western Canada at low resolution (10 km – left) and high resolution (2.5 km – right). The increase of resolution represents a factor of 4.

# SOLUTION:
	
### 1.	Libraries Used:
a.	I used Keras for Deep Learning and Theano as Backend
b.	Used DSSIMObjective from keras_contrib.losses. Later converted DSSIM to SSIM for score calculation.
(https://github.com/keras-team/keras-contrib)
c.	General Libraries: numpy, pandas, matplotlib, pickle, os, time


### 2.	Preparing to Load Data:

1. Generated a list of names of input_training data for training my Model. It includes 51 input_training files from ‘input_training_0000_0099.npy’ to ‘input_training_5000_5099.npy’.
2. Generated a list of names of input_training data for validation of my model. It includes 3 input_training files from ‘input_training_5100_5199.npy’ to ‘input_training_5300_5342.npy’.
3. Generated a list of 51 names of label_training data for target training of my model from ‘label_training_0000_0099.npy’ to ‘label_training_5000_5099.npy’. 
4. Generated a list of 3 label_training data for target validation of my model from ‘label_training_5100_5199.npy’ to ‘label_training_5300_5342.npy’.
5. Now I created a dictionary named ‘partition’ which stores all the above lists as partition['train_input'], partition['val_input'], partition['train_label'], partition['val_label'].

6. Created fuction named load_data which performs following work:
- np.load(): Loads the data
- np.transpose(): transposes the data for training in keras model
- standardize() : normalizes the data for better performance.


### 3.	Creating Score Functions:

It includes the following functions:
1. meanDifferenceCalculation()
2. minDifferenceCalculation()
3. maxDifferenceCalculation()
4. scoreCalculation():
- Creates a DSSIMObjective object with kernel_size=4
- Calculates DSSIM loss for images.
- Then calculates SSIM from DSSIM using formula:
ssim =  1 - ( 2 * dssim ).
- Calculates Mean Squared Error
- Calculates meanDiff, minDiff and maxDiff (a. b. and c.)
- Adds up all the scores to gives final score.



### 4.	Convolution Neural Network 

Here is the Architecture for my ConvNet Model in keras.


1. input_1 (InputLayer)         	(None, 256, 256, 15)   		   0        
2. conv1 (Conv2D)               	(None, 256, 256, 16)   	 	  2176      
3. bn_conv1 (BatchNormalization 	(None, 256, 256, 16)    	  64        
4. activation_1 (Activation)   	 (None, 256, 256, 16)   	   0         
5. conv2 (Conv2D)               	(None, 256, 256, 1)     	  145       
6. bn_conv2 (BatchNormalization 	(None, 256, 256, 1)      	 4         
7. activation_2 (Activation)    	(None, 256, 256, 1)      	 0         
**Total params: 2,389
Trainable params: 2,355
Non-trainable params: 34**

Optimization: optimizer='adam' , loss='mean_squared_error', metrics=scoreCalculation  (Function Built in Step 3. )

Explanation for fitting the model: As I was limited on computer resources I had to train each input file one by one. My system could not take the load of the memory of loading multiple files at once. Thus I was loading one file at a time, performing preprocessing and then fitting the file data in model.
Therefore I used a for loop for running 7 epochs and each epoch was training on 51 files and validating on 3 files. 
My validation split was of only 3 files because I wanted to mimic the train and test split size. As test data had around 248 entries thus I used 3 files for validation which has similar number of entries.   



### 5.	Plotting train and validation (score and loss)

Before plotting the results I wrote some simple scripts to extract training Loss and Score from saved history of loss and score.
 Then I Averaged up score/loss to give score/loss for 1 epoch as training score is for each file individually. So I averaged each batch of 51 scores to give score for 1 epoch. 
Same was done for Validation also where number of files were 3. 

 
![alt text](https://github.com/Nishant-Chhetri/Temperature_Forecast_Super_Resolution/blob/master/Results/Plots/train_loss.jpg)

![alt text](https://github.com/Nishant-Chhetri/Temperature_Forecast_Super_Resolution/blob/master/Results/Plots/train_score.jpg)

![alt text](https://github.com/Nishant-Chhetri/Temperature_Forecast_Super_Resolution/blob/master/Results/Plots/validation_loss.jpg)

![alt text](https://github.com/Nishant-Chhetri/Temperature_Forecast_Super_Resolution/blob/master/Results/Plots/validation_score.jpg)


 
	
### 6.	Results with test data

For evaluating each date and time entry individually, I used model.evaluate() for each record separately by reshaping each record to hold single entry/record ie. (1,256,256,15) and (1,256,256,1) and appended the loss and score to test_history list. 

![alt text](https://github.com/Nishant-Chhetri/Temperature_Forecast_Super_Resolution/blob/master/Results/Plots/test_loss_for_each_record.jpg)

![alt text](https://github.com/Nishant-Chhetri/Temperature_Forecast_Super_Resolution/blob/master/Results/Plots/test_score_for_each_record.jpg)
 
 
Then I stored each date and time record’s result to pandas test_results Dataframe having record for each date and time.  
Using test_resuts.describe() we can see that:

The average loss for test_results is  :  **average_loss=0.806500**

The average score for test_results is: **average_score=5.001534**

Results are stored in test_results.csv files which can be reproduced from given code. 

*Improvements: As I had limited computer resources, therefore could not built a deep enough Network like ResNet which product state of the art results for Super Resolution Problems. 
