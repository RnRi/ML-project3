"""
@author: Nicolas Angelard-Gontier
"""

import numpy as np
import csv


"""
Read the training cvs files and return a dictionary containing the training images as arrays.
@param option - if 1 (default) only grab unmodified training data,
    if 2 return original images and 180* rotated images,
    if 3 return original images, 90*, and 180* rotated images,
    if 4 return original images, 90*, 180*, and 270* rotated images.
@return train_inputs is of the form: {
    "image ID 1" : {
        "##"		: # <-- the number represented by this image.
        "0"  		: [...], <-- array of pixel values for unmodified test image.
        "90" 		: [...], <-- array of pixel values for the rotation by 90* to the right of the test image.
        "180"		: [...], <-- array of pixel values for the rotation by 180* of the test image.
        "270"		: [...] <-- array of pixel values for the rotation by 270* to the right of the test image.
    },
    "image ID 2" : {
        ...
    },
    ...
}
"""
def getTrainData(option=1):
    KAGGLE_TRAIN_IN = "data_and_scripts/train_inputs.csv"
    KAGGLE_TRAIN_OUT = "data_and_scripts/train_outputs.csv"

    # Load all training inputs
    print "loading the train data..."
    train = {}
    with open(KAGGLE_TRAIN_IN, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the header
        for train_input in reader:
            train[train_input[0]] = {} # for each image, we have a dictionary.
            train_input_no_id = []
            for dimension in train_input[1:]:
                train_input_no_id.append(float(dimension))
            train[train_input[0]]['0'] = np.asarray(train_input_no_id) # Load each sample as a numpy array.
            if option == 2 or option == 3 or option == 4:
                train[train_input[0]]['180'] = get180(train_input_no_id)
            if option == 3 or option == 4:
                train[train_input[0]]['90'] = get90(train_input_no_id, 48, 48)
            if option == 4:
                train[train_input[0]]['270'] = get270(train_input_no_id, 48, 48)

    # Load all training ouputs
    print "loading training outputs..."
    with open(KAGGLE_TRAIN_OUT, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the header
        for train_output in reader:  
            train_output_no_id = int(train_output[1])
            train[train_output[0]]['##'] = train_output_no_id

    print "done loading train data."
    return train

"""
Read the test csv file and returns a dictionary containing the test images as arrays.
@param option - if 1 (default) only grab unmodified testing data,
    if 2 return original images and 180* rotated images,
    if 3 return original images, 90*, and 180* rotated images,
    if 4 return original images, 90*, 180*, and 270* rotated images.
@return test_inputs is of the form: {
    "image ID 1" : {
        "0"         : [...], <-- array of pixel values for unmodified test image.
        "90"        : [...], <-- array of pixel values for the rotation by 90* to the right of the test image.
        "180"       : [...], <-- array of pixel values for the rotation by 180* of the test image.
        "270"       : [...], <-- array of pixel values for the rotation by 270* to the right of the test image.
    },
    "image ID 2" : {
        ...
    },
    ...
}
"""
def getTestData(option=1):
    KAGGLE_TEST = "data_and_scripts/test_inputs.csv"
    print "loading test data..."
    # Load all test inputs
    test_inputs = {}
    with open(KAGGLE_TEST, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the header
        for test_input in reader:
            test_inputs[test_input[0]] = {} # for each image, we have a dictionary.
            test_input_no_id = []
            for pixel in test_input[1:]: # Start at index 1 to skip the Id
                test_input_no_id.append(float(pixel))
            test_inputs[test_input[0]]['0'] = np.asarray(test_input_no_id) # Load each sample as a numpy array.
            if option == 2 or option == 3 or option == 4:
                test_inputs[test_input[0]]['180'] = get180(test_input_no_id)
            if option == 3 or option == 4:
                test_inputs[test_input[0]]['90'] = get90(test_input_no_id, 48, 48)
            if option == 4:
                test_inputs[test_input[0]]['270'] = get270(test_input_no_id, 48, 48)

    print "done loading test data."
    return test_inputs

"""
Returns the image array rotated by 180*.
@param image_array - the array pixel values for the image.
@return the reversed of the given array.
"""
def get180 (image_array):
	return np.asarray(image_array[::-1])

"""
Returns the image array rotated by 90* to the right.
@param image_array - the array pixel values for the image.
@param image_width - the number of 'columns' of pixels in the image.
@param image_height - the number of 'lines' of pixels in the image.
@return the array corresponding to the 90* rotated image.
"""
def get90 (image_array, image_width, image_height):
	a = []
	image = np.asarray(image_array).reshape(image_height, image_width)
	for j in range(image_width):
		for i in range(image_height)[::-1]:
			a.append(image[i][j])
	return np.asarray(a)


"""
Returns the image array rotated by 270* to the right.
@param image_array - the array pixel values for the image.
@param image_width - the number of 'columns' of pixels in the image.
@param image_height - the number of 'lines' of pixels in the image.
@return the array corresponding to the 270* rotated image (90+180).
"""
def get270 (image_array, image_width, image_height):
	return np.asarray(get180(get90(image_array, image_width, image_height)))


"""
uncomment to test
"""
#image = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
#print get180(image)
#print get90(image, 4, 4)
#print get270(image, 4, 4)


