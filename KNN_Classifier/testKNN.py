import random
from knn import *
# from naive_bayes import *
from random import shuffle
from math import pow
from sklearn.model_selection import KFold

def loadData(filename, split, trainingSet, testSet, trainingSetY, testSetY):

	with open(filename, 'rb') as file:
		lines = file.read().split('\n')
		lines = [l.split(',') for l in lines]
		dataset = []

		for line in lines:
			curr = []
			for x in line:
				if x != line[-1]:
					curr.append(float(x))
				else:
					curr.append(x)
			dataset.append(curr)
		
		for x in range(len(dataset)-1):
			for y in range(34):
				if random.random() < split:
					trainingSet.append(dataset[x][:34])
					trainingSetY.append(dataset[x][35])
	        	else:
	 				testSet.append(dataset[x][:34])
	 				testSetY.append(dataset[x][35])
	 		
	 		
if __name__ == "__main__":

	trainingSet = []
	testSet = []
	trainingSetY = []
	testSetY = []
	split = 0.67
	loadData('./testData.txt', split, trainingSet, testSet, trainingSetY, testSetY)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))

	knn = KthNearestNeighbour(trainingSet, trainingSetY, testSet)

	#choosing k = N^(1/2)
	k = int(pow(len(trainingSet), 0.5))
	print 'chose value k =', k
	predictions = knn.predict(k)

	for x in range(len(predictions)):
		print 'predicted=', predictions[x], ', actual=', testSetY[x]
	print knn.getValidationAccuracy(predictions, testSetY)

