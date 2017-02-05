####################################################################
# This code is CNN based face recognition programme. It reads 5
# faces from ORL database and the rest 5 are used as test.
# PCA_Performance shows the recognition performance. 
#  
# Download the ORL database from internet. 
# This code is written by Saurabh Puri in order to show the face
# recognition task
#######################################################################
import keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

def baseline_model():
    #creat model
    model = Sequential()    
    model.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=(1,w,h)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))    
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))    
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#fix the random seed for reproducibility
seed = 7
np.random.seed(seed)

zz=1;
noc=40;                 #no_of_classes
nots=5;                 #no_of_training_set

#width and height is hardcoded but could be derived from the image itself
#sometimes for a better performance, cropping of image is required as PCA is generally very sensitive to variations in the image (like light, shadow, etc.)
w = 112
h = 92

#Split the dataset into training and test set
#Folder location: ./att_faces/s*/*.pgm
#First half images in each class is considered as training set and other half are considered to be test set
X_train = np.empty(w*h, dtype=np.float32)
y_train = np.empty(1, dtype=np.int32)
X_test = np.empty(w*h, dtype=np.float32)
y_test = np.empty(1, dtype=np.int32)
for i in range(1,noc+1):
    for j in range(1,nots+1):
        #print(str(i) +' '+ str(j))
        file= "./att_faces/s" + str(i) + "/" +  str(j) + ".pgm"
        im = cv2.imread(file)        
        im = im.transpose((2,0,1))
        im = np.expand_dims(im,axis=0)
        imgray = im[0][0]
        im1D = imgray.flatten('F')
        X_train = np.vstack((X_train,im1D))
        y_train = np.hstack((y_train,i-1))

for i in range(1,noc+1):
    for j in range(nots+1,nots+6):
        #print(str(i) +' '+ str(j))
        file= "./att_faces/s" + str(i) + "/" +  str(j) + ".pgm"
        im = cv2.imread(file)   
        im = im.transpose((2,0,1))
        im = np.expand_dims(im,axis=0)
        imgray = im[0][0]
        im1D = imgray.flatten('F')
        X_test = np.vstack((X_test,im1D))
        y_test = np.hstack((y_test,i-1))

#delete first row as it was empty
X_train = np.delete(X_train,(0),axis=0)
y_train = np.delete(y_train,(0),axis=0)
X_test = np.delete(X_test,(0),axis=0)
y_test = np.delete(y_test,(0),axis=0)

print('loaded')

print(X_train.shape[0])
X_train = X_train.reshape(X_train.shape[0], 1, 112, 92).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 112, 92).astype('float32')

print(X_train.shape)

#normalize to 0-1
X_train = X_train/255
X_test = X_test/255

#one-hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

print(num_classes)

#build the model
model = baseline_model()

#Fit the model
history1 = model.fit(X_train,y_train, validation_data=(X_test,y_test), nb_epoch=50, batch_size=8, verbose=1)

print('Prediction on saved sample:')
#print(str(model.predict(X_test)))
out = model.predict(X_test[:1])
print(np.argmax(out))
#print(y_test[:200])


