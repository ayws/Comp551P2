'''
	Assignment 2
	written for COMP-551 (Applied Machine Learning)
'''

import numpy as np 
import codecs
from math import sqrt
from operator import itemgetter
from sklearn.model_selection import KFold
from random import seed, randrange

class KthNearestNeighbour:

	def __init__(self, train_X, train_Y, test_X):
		self.train_X = train_X
		self.train_Y = train_Y
		self.test_X = test_X

		if (len(train_X) != len(train_Y)): 
			print 'X and Y need to be of same size.'
			exit()

	#straight-line distance of the path connecting two points
	# x and y are both lists
	def __euclideanDist(self, x, y):
		return sqrt(sum(pow(x_i - y_i, 2) for x_i, y_i in zip(x,y)))

	'''
	parameters: x = a single datapoint vector, k = the number of neighbours
	looks at the Euclidean distance for each data point, finds its k-nearest neighbours.
	The point is then classified into Whichever class is most frequent among the neighbours.
	'''
	def __findNeighbours(self, x, k):

		# print 'finding', k, 'nearest neighbours'

		knn = []
		distances = []

		# for x in X, find the k nearest training samples to x
		for i in range(len(self.train_X)):
			curr_dist = self.__euclideanDist(x, self.train_X[i])
			distances.append((curr_dist, self.train_X[i], self.train_Y[i]))
		
		distances.sort(key=itemgetter(0)) #sort by distance
		
		
		#get k nearest training samples 
		for n in range(0,k):
			knn.append((distances[n][1], distances[n][2])) #append point + its classification

		return knn


	'''
	Takes as parameter a list of k-nearest neighbours for a data point.
	'''
	def __predictPoint(self, knn):

		# print 'predicting a single data point...'

		classCount = {}

		for (x,y) in knn:
			classif = y
			if classif in classCount:
				classCount[classif] += 1
			else:
				classCount.update({classif:1})
		
		#sort all classes by their number of associated classifications
		# the prediction is the top classification of the sorted results
		sortedClass = sorted(classCount.iteritems(), reverse=True, key=lambda (k,v):(v,k))
		return sortedClass[0][0]

	def getValidationAccuracy(self, predictions, validationSet=None):

		if not validationSet: validationSet = self.train_Y #1-fold validation

		correct = 0
		for y in range(len(validationSet)):
			if(validationSet[y] == predictions[y]):
				correct += 1
		return (float(correct) / len(validationSet)) * 100.0

	def predict(self, k=3):

		print 'predicting data points'

		predictions = []
		for test_x in self.test_X:

			neighbours = self.__findNeighbours(test_x, k)
			classif = self.__predictPoint(neighbours)
			predictions.append(classif)
			# print 'CLASSIFICATION', classif
			
		return predictions
		
