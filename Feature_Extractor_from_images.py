#%%

# -*- coding: utf-8 -*-

#Author - Ankit Dwivedi

import datetime as dt
start_time = dt.datetime.now()

from os import listdir
from pickle import dump
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
#%%
'''Below is a function named extract_features() that, given a directory name,
will load each photo, prepare it for VGG, and collect the predicted features
from the VGG model. The image features are a 1-dimensional 4,096 element vector'''


#%%
# extract features from each photo in the directory
def extract_features(directory):
	# load the model
	# load the model
   model = Sequential()
   # input: 224x224 images with 3 channels -> (224, 224, 3) tensors.
   # this applies 32 convolution filters of size 3x3 each.
   model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
   model.add(Conv2D(32, (3, 3), activation='relu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))

   model.add(Conv2D(64, (3, 3), activation='relu'))
   model.add(Conv2D(64, (3, 3), activation='relu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
   model.add(Dropout(0.25))

   #model.add(Flatten())
   model.add(Dense(4096, activation='relu'))
   model.add(Dropout(0.5))
   model.add(Dense(10, activation='softmax'))
	# re-structure the model
   model.layers.pop()
   model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	# summarize
   print(model.summary())
	# extract features from each photo
   features = dict()
   for name in listdir(directory): #The method listdir() returns a list containing the names of the entries in the directory given by path.
		# load an image from file
	   filename = directory + '/' + name
	   image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
	   image = img_to_array(image)
		# reshape data for the model
	   image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	   #image = preprocess_input(image)
		# get features
	   feature = model.predict(image, verbose=0)
		# get image id
	   image_id = name.split('.')[0]
		# store feature
	   features[image_id] = feature
	   print('>%s' % name)
   return features

# extract features from all images
directory = 'Sample_Data'

features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))
# the extracted features stored in ‘features.pkl‘ for later use

run_time = dt.datetime.now() - start_time
print ("Total job runnning time is {}".format(run_time))