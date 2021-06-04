import pandas as pd
import random
import math
import operator
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn import metrics

def loadDataset(filename, split):
    trainingSet = []
    testSet = []
    df = pd.read_csv(filename, header=None)
    array = df.to_numpy()
    # : will get all rows, :-1 will get all columns except last column
    trainingSet = array[:, :-1]
    # : will get all rows, 4 will get only the last column
    testSet = array[:, 4]
    # train_test_split method will split the trainingSet 
    # and testSet into features and lables. Both trainingSet
    # and testSet will auto shuffled. test_size is the split 
    # percentage that will go to the test sets.
    features_train, features_test, labels_train, labels_test = train_test_split(trainingSet, testSet, test_size = split)
    return features_train, features_test, labels_train, labels_test


# Writing accuracy finder since we could not figure out how to convert the
# labels to float.As of writing not finished (this sentence will be deleted
# when finished)
# 
# How it should work:
# When findAccuracy is called, the data will already be inside the sets.
# Every row of feature_test would be compared to every of row of feature_train
# If a row of feature_test matched a row in feature_train then the label
# associated with that row in feature_train would be gotten and compared
# to the prediction lable associated with the feature_test row.
# If the predicated label was the exact same as the correct label then
# a counter for how many labels were correctly guessed would incremeted
# At the end, the accuracy would be found by doing the total correct/51
def findAccuracy(features_train, features_test, labels_train, prediction):
    accNum = 0
    ftTestL = len(features_test) -1
    ftTrainL = len(features_train) -1
    for x in range(ftTestL):
        for y in range(ftTrainL):
            if features_test[x][0] == features_train[y][0]:
                if features_test[x][1] == features_train[y][1]:
                    if features_test[x][2] == features_train[y][2]:
                        if features_test[x][3] == features_train[y][3]:
                                correctLb = labels_train[y]
                                if prediction[x] == correctLb:
                                    accNum+=1
    accuracy = accNum/51
    return accuracy
    
    
    
def main():
    #BIG NOTE: features_test = 99, features_train = 51. These values are
    #reversed and should be swapped with each other. 
    #Lazy fix: switched the names calling loadDataset
    
    #should be 2d array with 150 rows and 4 columns representing features
    trainingSet=[]
    #should be 1d with 150 labels represented in a number format (Ex: 0 = setsona)
    testSet=[]
    url = 'https://raw.githubusercontent.com/ruiwu1990/CSCI_4120/master/KNN/iris.data'
    # features_train and features_lables represents the training dataset 
    # of features and their correct lables that will be used by the machine
    # for making predictions and checking accuracy.
    #
    # features_test and lables_test represent the datasets that will be used
    # to test the machine
    features_test, features_train, labels_test, labels_train = loadDataset(url, 0.66)
    
    # for loop to iterate from 1 to 20 (commented out as rest of code
    # is not fully finished. The loop will encompass everything below).
    # range_of_k = range(1, 20)
    # for k in range_of_k:
    
    
    # Euclidean distance calculation is built into KNeighborsClassifier's
    # class parameters and is set to it by default so only n_neighbors needs
    # to be changed
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(features_train, labels_train)
    
    #print(features_test)
    
    # using kneighbors mehthod to find neighbors
    # To read when printed, the values inside [] represent
    # the row which is the neighbor. Ex: [[24]] means row 0's
    # neighbor is row 24
    #print(neigh.kneighbors(features_test, return_distance = False))
    neigh.kneighbors(features_test, return_distance = False)
    
    # use predict method
    prediction = neigh.predict(features_test)
    #print(features_test)
    #print(prediction)
    
    
    # score method used for accuracy
    # currently, does not work properly as it repeats calculated accuracy numbers
    # could be something to do with the neighbors not being known for features_test
    accuracy = findAccuracy(features_train, features_test, labels_train, prediction)
    print(accuracy)
   

main()
