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
def findAccuracy(features_train, features_test, labels_train, labels_test):
    accNum = 0
    for x in range(len(features_test)):
        for y in range(len(features_train)):
            if features_test[x] == features_train[y]:
                correctLb = labels_train[y]
                if labels_test[x] == correctLb:
                    accNum+=1
    accuracy = (accNum/51)*100
    return accuracy
    
    
    
def main():
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
    features_train, features_test, labels_train, labels_test = loadDataset(url, 0.66)
    
    
    # Euclidean distance calculation is built into KNeighborsClassifier's
    # class parameters and is set to it by default so only n_neighbors needs
    # to be changed
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(features_train, labels_train)
    
    #print(features_test)
    
    # using kneighbors mehthod to find neighbors
    # To read when printed, the values inside [] represent
    # the row which is the neighbor. Ex: [[24]] means row 0's
    # neighbor is row 24
    #print(neigh.kneighbors(features_test, return_distance = False))
    
    #use predict method
    prediction = neigh.predict(features_test)
    #print(features_test)
    #print(prediction)
    
    
    #score method used for accuracy
    #error produced with findAccuracy, The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
    accuracy = findAccuracy(features_train, features_test, labels_train, labels_test)
    print(accuracy)

main()
