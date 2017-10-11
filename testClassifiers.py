import csv
import random
from knn import *
from random import shuffle

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    shuffle(dataset)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])

def getAccuracy(testY, predictions):
	correct = 0
	for x in range(len(testSet)):
		if (testY[x] == predictions[x]): 
			correct += 1
	return (correct/float(len(testSet))) * 100.0


if __name__ == "__main__":

	trainingSet = []
	testSet = []
	split = 0.67
	loadDataset('./data/test_dataset.csv', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))

	trainingSetY = []
	testSetY = []
	for row in trainingSet:
		trainingSetY.append(row[-1])
		del row[-1]

	for row in testSet:
		testSetY.append(row[-1])
		del row[-1]

	knn = KthNearestNeighbour(trainingSet, trainingSetY, testSet)
	predictions = knn.predict()

	print 'PREDICTIONS:', predictions
	print getAccuracy(testSetY, predictions)
