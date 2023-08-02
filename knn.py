import os
import cv2
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
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


neighbour = int(input("Please enter your k value: "))
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

# Loading our model
model = KNeighborsClassifier(n_neighbors=neighbour, n_jobs=4)

# Splitting our data into subsets, %80 for training and %20 for testing
trainX, testX, trainY, testY = train_test_split(data_train, labels, test_size=0.2, random_state=42)

# Training our model
model.fit(trainX, trainY)

# Testing our model
pred = model.predict(testX)

# Printing a classification report to get accuracy, precision, recall and f1 scores.
print(classification_report(testY, pred))
