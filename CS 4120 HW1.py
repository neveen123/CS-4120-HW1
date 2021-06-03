import pandas as pd
import random
import math
import operator
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def loadDataset(filename, split):
    trainingSet = []
    testSet = []
    df = pd.read_csv(filename, header=None)
    array = df.to_numpy()
    random.shuffle(array)
    training_len = int(len(array)*split)
    trainingSet = array[:training_len]
    testSet = array[training_len:]
    return trainingSet, testSet



def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


def main():
    trainingSet=[]
    testSet=[]
    predications = []
    split = 0.67
    url = 'https://raw.githubusercontent.com/ruiwu1990/CSCI_4120/master/KNN/iris.data'
    trainingSet, testSet = loadDataset(url, 0.66)
    
    neigh = KNeighborsClassifier(n_neighbors=1)
    
    #use predict method
    #print(neigh.predict())
    
    #using kneighbors mehthod to find neighbors
    #print(neigh.kneighbors(testSet[1], return_distance = False))
    
    #score method used for accuracy
    #score(testSet, trainingSet, sample_weight = None)

main()
