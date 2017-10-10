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

		# if (len(train_X) != len(train_Y)): 
			# print 'X and Y need to be of same size.'
			# exit()

	#straight-line distance of the path connecting two points
	# x and y are both lists
	def euclideanDist(self, x, y):
		return sqrt(sum(pow(x_i - y_i, 2) for x_i, y_i in zip(x,y)))

	'''
	parameters: train_set_Y, train_set_Y, test_set_X, k
	looks at the Euclidean distance for each data point, finds its k-nearest neighbours.
	The point is then classified into Whichever class is most frequent among the neighbours.
	'''
	def findNeighbours(self, x, k):

		print 'finding', k, 'nearest neighbours'

		knn = []
		distances = []

		# for x in X, find the k nearest training samples to x
		for i in range(len(self.train_X)):
			curr_dist = self.euclideanDist(x, self.train_X[i])
			distances.append((curr_dist, self.train_X[i], self.train_Y[i]))
		
		distances.sort(key=itemgetter(0)) #sort by distance
		
		print distances[:5]
		
		#get k nearest training samples 
		for neighbour in range(0,k):
			knn.append((distances[neighbour][1], distances[neighbour][2])) #append point + its classification

		return knn


	'''
	Takes as parameter a list of k-nearest neighbours for a data point.
	'''
	def predictPoint(self, knn):

		print 'predicting a single data point...'

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

	#for tuning the k hyperparameter
	def cv_split(self, k):

		k_fold_size = int(len(self.train_X) / k)
		X_copy = self.train_X.tolist()
		Y_copy = self.train_Y.tolist()
		k_folds_X = []
		k_folds_Y = []

		for curr_k in range(k):
			curr_fold_X = []
			curr_fold_Y = []
			while len(curr_fold_X) < k_fold_size:
				i = randrange(len(X_copy))
				j = randrange(len(Y_copy))
				curr_fold_X.append(X_copy.pop(i))
				curr_fold_Y.append(Y_copy.pop(j))
			k_folds_X.append(curr_fold_X)
			k_folds_Y.append(curr_fold_Y)

		return k_folds_X, k_folds_Y


	def predict(self):

		print 'training KNN classifier...'

		seed(1) #seeds randomization to ensure same split for every execution
		# k_folds_X, k_folds_Y = self.cv_split(3)

		# k_predictions = []
		# for k in range(len(k_fold_X)):

		# 	predictions = []
		# 	for test_x in self.test_X:

		# 		neighbours = self.findNeighbours(3)
		# 		classif = self.predictPoint(neighbours)
		# 		predictions.append(classif)
			
		# 	print predictions
		# 	k_predictions = k_predictions + predictions
		# return k_predictions



		predictions = []
		for test_x in self.test_X:

			print test_x

			neighbours = self.findNeighbours(test_x, 3)
			classif = self.predictPoint(neighbours)
			predictions.append(classif)
			
		print predictions
		
