# BIG NOTE: features_test = 99, features_train = 51. These values are
# reversed and should be swapped with each other. 
# Lazy fix: switched the names calling loadDataset
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

    
def main():
    
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
    range_of_k = range(1, 21)
    for k in range_of_k:
        
        # Euclidean distance calculation is built into KNeighborsClassifier's
        # class parameters and is set to it by default so only n_neighbors needs
        # to be changed
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(features_train, labels_train)
        
        # using kneighbors mehthod to find neighbors
        # To read when printed, the values inside [] represent
        # the row which is the neighbor. Ex: [[24]] means row 0's
        # neighbor is row 24
        #print(neigh.kneighbors(features_test, return_distance = False))
        neigh.kneighbors(features_test, return_distance = False)
        
        # use predict method
        prediction = neigh.predict(features_test)
        
        # score method used for accuracy
        # works but may not be correct parameters to use
        s = neigh.score(features_test, labels_test, sample_weight = None)
        print(float(s))
        print(k)

main()
