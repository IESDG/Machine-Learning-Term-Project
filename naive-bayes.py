import os
import cv2
import numpy as np
from imutils import paths
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


def createData(imagePaths):
    # Variables to hold our data and our labels
    data = []
    labels = []
    interval = 100
    # Main loop to loop over every image in our input file
    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        # file/image -> file is the label of the image
        label = imagePath.split(os.path.sep)[-2]
        data.append(image)
        labels.append(label)

        # A print function to show the progress of creating data
        if interval > 0 and i > 0 and (i + 1) % interval == 0:
            print("{}/{} processed".format(i + 1, len(imagePaths)))

    return np.array(data), np.array(labels)


#------------------------- !!! Main Data Path !!! -----------------------------------------------------------------------
# train -> 67.000+ images
# test -> 22.000+ images
# train2 and test2 -> 1.000 images
# Using train can eat up all the ram in the computer, using test ( around %20 of the total database ) is preferred
data_train_path = list(paths.list_images('train'))

#-----------------------------------------------------------------------------------------------------------------------
data_train, labels = createData(data_train_path)

# Encoding the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# Reshaping our data from (x, 100, 100, 3) to (x,30000) [100x100 pixels and 3 is for rgb]
data_train = data_train.reshape((data_train.shape[0], 100 * 100 * 3))

# Splitting our data into subsets, %80 for training and %20 for testing
trainX, testX, trainY, testY = train_test_split(data_train, labels, test_size=0.2, random_state=42)

# Loading our model -> Naive bayes using GaussianNB
tester = GaussianNB()

# Training our model
print(tester.fit(trainX, trainY))

# Testing our model
print("Please wait")
pred = tester.predict(testX)

# Printing a classification report to get the information.
report = classification_report(testY, pred)
print(report)

