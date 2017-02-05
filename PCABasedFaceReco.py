####################################################################
# This code is PCA base face recognition programme. It reads 5
# faces from ORL database and the rest 5 are used as test.
# PCA_Performance shows the recognition performance. 
#  
# Download the ORL database from internet. 
# This code was modified by Saurabh Puri in order to show the face
# recognition task
#######################################################################
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import cv2

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

#normalize to 0-1
X_train = X_train/255
X_test = X_test/255

# initiate PCA and fit to the training data
pca = PCA(n_components=40)
pca.fit(X_train)

# transform
X_transformed = pca.transform(X_train)
newdata_transformed = pca.transform(X_test)

#initiate a classifier and then fit eigen faces and labels
clf = SVC()
clf.fit(X_transformed,y_train)

# predict new labels using the trained classifier
pred_labels = clf.predict(newdata_transformed)

#output the accuracy_score
score = accuracy_score(y_test,pred_labels,True)
print(score)

##Print the predicted labels
#print(pred_labels)
