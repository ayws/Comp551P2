import csv
import random
from knn import *
# from naive_bayes import *
from random import shuffle
from math import pow
from sklearn.model_selection import KFold

def loadDataset(filename, split, trainingSet, testSet):

	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])


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

	#choosing k = N^(1/2)
	k = int(pow(len(trainingSet), 0.5))
	print 'chose value k =', k
	predictions = knn.predict(k)

	for x in range(len(predictions)):
		print 'predicted=', predictions[x], ', actual=', testSetY[x]
	print knn.getValidationAccuracy(predictions, testSetY)

	
