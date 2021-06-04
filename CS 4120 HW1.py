import pandas as pd
import random
import math
import operator
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn import metrics

def loadDataset(filename, split):
    trainingSet = []
    testSet = []
    #features_train = []
    #features_test = []
    #lables_train = []
    #labels_test = []
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
    features_train, features_test, lables_train, labels_test = train_test_split(trainingSet, testSet, test_size = split)
    
    return features_train, features_test, lables_train, labels_test


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
    features_train, features_test, lables_train, labels_test = loadDataset(url, 0.66)
    
    
    
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(features_train, lables_train)
    
    print(features_test)
    
    #use predict method
    prediction = neigh.predict(features_test)
    
    print(prediction)
    
    #using kneighbors mehthod to find neighbors
    #print(neigh.kneighbors(testSet, return_distance = False))
    
    #score method used for accuracy
    #score(testSet, trainingSet, sample_weight = None)

main()
