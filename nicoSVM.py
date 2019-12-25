#!/usr/bin/env python

"""
@author: Nicolas Angelard-Gontier
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from scipy import ndimage
from datetime import datetime


def save_obj(obj, name):
    with open('obj/'+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/'+name+'.pkl', 'rb') as f:
        return pickle.load(f)


"""
Shows a given image.
@param image - the image in a 1D array.
@param dim - the dimension of the image.
"""
def showImage(image, dim):
    image = image.reshape(dim,dim)
    plt.imshow(image, cmap="Greys_r")
    plt.show()


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ['0','1','2','3','4','5','6','7','8','9'])
    plt.yticks(tick_marks, ['0','1','2','3','4','5','6','7','8','9'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"""
This methods applies thresholding to a given image array.
@param image_array - the array of pixel values corresponding to the image.
@param threshold - the threshold value that decides if the pixel is going to be black or white.
@return - the modified image.
"""
def binaryImage(image_array, threshold=0.5):
    for i in range(len(image_array)):
        if image_array[i] <= 0.5:
            image_array[i] = 0.0
        else:
            image_array[i] = 1.0
    return image_array


# Uncomment the 1st time to generate obj/train.pkl & obj/test.pkl
# No need after that, can directly load the data from the pkl files.
"""
from nicoLoadTestData import getTestData, getTrainData

test = getTestData(option=4)
print len(test)
print test["1994"]

train = getTrainData(option=4)
print len(train)
print train["2015"]

print "saving test..."
save_obj(test, "test")
<F4>
print "saving train..."
save_obj(train, "train")
"""


#####################################################
################ SVM on regular data ################
#####################################################
print "fetching train..."
TRAIN = load_obj("train")

#print "fetching test..."
#TEST = load_obj("test")

TRAIN_SIZE = 8000
TEST_SIZE = 2000
DO_PCA = [100,225,400,784,900,1125,2304] # try with 10*10, 15*15, 20*20, 28*28, 30*30, 35*35, 48*48
GAUSSIAN_FILTER = [1,2,2.5,3,3.5,4]      # try with 1, 2, 2.5, 3, 3.5, 4
BINARY_IMAGE = False

K = 8 # number of versions of the training data.
#if GAUSSIAN_FILTER > 0:
#    K += 4
#if BINARY_IMAGE:
#    K += 4

"""
Generating X and Y matrices.
"""
def generateXY(pca, gauss):
    X = []
    Y = []
    #global train
    #global TRAIN_SIZE
    #global TEST_SIZE
    #global BINARY_IMAGE
    
    print "generating X & Y, preprocessing..."
    for image in TRAIN.values()[:TRAIN_SIZE+TEST_SIZE]: #max: 50000
        X.append(image["0"])   # train on original image.
        X.append(image["90"])  # train on 90* rotated image.
        X.append(image["180"]) # train on 180* rotated image.
        X.append(image["270"]) # train on 270* rotated image.
        Y.extend([image["##"]]*4) # add the target value 4 times.
        if BINARY_IMAGE:
            X.append(binaryImage(image["0"]))   # train on binary of original.
            X.append(binaryImage(image["90"]))  # train on binary of 90* rotated.
            X.append(binaryImage(image["180"])) # train on binary of 180* rotated.
            X.append(binaryImage(image["270"])) # train on binary of 270* rotated.
            Y.extend([image["##"]]*4) # add the target value 4 times.
        if gauss > 0:
            X.append( ndimage.gaussian_filter(image["0"].reshape(48,48), gauss).flatten() )
            X.append( ndimage.gaussian_filter(image["90"].reshape(48,48), gauss).flatten() )
            X.append( ndimage.gaussian_filter(image["180"].reshape(48,48), gauss).flatten() )
            X.append( ndimage.gaussian_filter(image["270"].reshape(48,48), gauss).flatten() )
            Y.extend([image["##"]]*4)
    X = np.asarray(X)
    Y = np.asarray(Y)
    assert X.shape == (K*(TRAIN_SIZE+TEST_SIZE), 2304)
    assert Y.shape == (K*(TRAIN_SIZE+TEST_SIZE), )

    if pca > 0 and pca <= 2304:
        p = PCA(n_components=pca)
        X = p.fit_transform(X)
        assert X.shape == (K*(TRAIN_SIZE+TEST_SIZE), pca)
    print X.shape

    if not X.flags['C_CONTIGUOUS']:
        print "WARNING: X is not C-ordered contiguous."
    if not Y.flags['C_CONTIGUOUS']:
        print "WARNING: Y is not C-ordered contiguous."

    return X,Y

"""
Training classifier.
"""
def train(alg, c, gamma, coef, d):
    print "training the classifier..."
    return svm.SVC(
        cache_size=1000,
        kernel=alg,   # try 'rbf', 'poly', 'sigmoid'
        C=c,            # try 0.1, 0.5, 1, 5, 10
        gamma=gamma,    # try 0.1, 0.5, 1, 5, 10
        coef0=coef,      # try 0.0, 1.0, 5, 10
        degree=d,     # try 2.0, 3.0, 4, 5
    ).fit(X[:TRAIN_SIZE*K],Y[:TRAIN_SIZE*K]) # train on the first TRAIN_SIZE points.


"""
Making predictions & write to file.
"""
def predict(classifier, X, Y):
    #global TEST_SIZE
    #global TRAIN_SIZE
    #global K
    print "making predictions..."

    y_test = []
    y_pred = []

    correct = 0.0
    for i in range(0, TEST_SIZE*K, K): # i goes form 0 to TEST_SIZE*K by steps of K
        predictions = []
        for j in range(K): # j goes from 0 to K
            p = classifier.predict(X[(TRAIN_SIZE*K)+i+j])[0]
            predictions.append(p)
        counts = np.bincount(predictions)
        prediction = np.argmax(counts)

        """Uncomment to see the image and the predictions"""
        #for p in predictions:
        #    print " ~", p
        #print " correct:", Y[(TRAIN_SIZE*4)+i]
        #for j in range(K):
        #    if DO_PCA > 0 and DO_PCA <= 2304:
        #        showImage(X[(TRAIN_SIZE*K)+i+j], np.sqrt(DO_PCA))
        #    else:
        #        showImage(X[(TRAIN_SIZE*K)+i+j], 48)

        if prediction == Y[(TRAIN_SIZE*K)+i]: # take the most popular prediction.
            correct += 1.0

        y_test.append(Y[(TRAIN_SIZE*K)+i])
        y_pred.append(prediction)

    print correct / TEST_SIZE
    return y_test, y_pred


#for pca in DO_PCA:
#    for gauss in GAUSSIAN_FILTER:
pca = 225
gauss = 2

start = datetime.now()
print "pca:%f gauss:%f" % (pca, gauss)
X,Y = generateXY(pca, gauss)

#CGAMMA = [0.1,0.5,1,5,10]
#COEF = [0,1,5,10]
#DEG = [2,3,4,5]
#for c in CGAMMA:
#    for gamma in CGAMMA:
#        for d in DEG:
#            for coef in COEF:
c=1.0
gamma=0.5
d=2
coef=1

print "POLY ; C=%f ; gamma=%f ; degree=%d ; coef=%d" % (c, gamma, d, coef)
classifier = train('poly', c, gamma, coef, d)
y_test, y_pred = predict(classifier, X, Y)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()


"""
Uncomment to write to file
"""
#test_output_file = open('data_and_scripts/test_output.csv', "wb")
#writer = csv.writer(test_output_file, delimiter=',') 
#writer.writerow(['Id', 'Prediction']) # write header
#for idx in TEST.keys():
#    prediction1 = classifier.predict(TEST[idx]["0"])[0]   # predict on the original image
#    prediction2 = classifier.predict(TEST[idx]["90"])[0]  # predict on the 90* rotated image
#    prediction3 = classifier.predict(TEST[idx]["180"])[0] # predict on the 180* rotated image
#    prediction4 = classifier.predict(TEST[idx]["270"])[0] # predict on the 270* rotated image
#    prediction5 = classifier.predict(
#    	ndimage.gaussian_filter(TEST[idx]["0"].reshape(48,48), gauss).flatten()
#    )[0]
#    prediction6 = classifier.predict(
#    	ndimage.gaussian_filter(TEST[idx]["90"].reshape(48,48), gauss).flatten()
#    )[0]
#    prediction7 = classifier.predict(
#    	ndimage.gaussian_filter(TEST[idx]["180"].reshape(48,48), gauss).flatten()
#    )[0]
#    prediction8 = classifier.predict(
#    	ndimage.gaussian_filter(TEST[idx]["270"].reshape(48,48), gauss).flatten()
#    )[0]
#
#    counts = np.bincount([prediction1,prediction2,prediction3,prediction4,
#    	prediction5,prediction6,prediction7,prediction8])
#    prediction = np.argmax(counts)
#
#    row = [idx, prediction]
#    writer.writerow(row)
#test_output_file.close()


print datetime.now() - start
print "-----------------------------------------"

