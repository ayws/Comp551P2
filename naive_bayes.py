import numpy as np 
from math import sqrt, exp, pow
import copy
from operator import add
from itertools import izip

class MultinomialNaiveBayes:

	def __init__(self, train_X, train_Y, test_X, smoothingParam=1.0):
		self.train_X = train_X
		self.train_Y = train_Y
		self.test_X = test_X
		self.smoothingParam = smoothingParam
		self.ClassifDict = self.__groupByClass()
		self.log_prob_by_class = self.__calc_class_prob()


	def __groupByClass(self):

		ClassifDict = {}
		for row in range(len(self.train_X)):
			vect = self.train_X[row]
			classif = self.train_Y[row]

			if (classif in ClassifDict):
				ClassifDict[classif].append(vect)
			else: 
				ClassifDict[classif] = [vect]

		return ClassifDict

	# the prior probability for a class
	# calculate P(y)
	# returns a list with the log probabiltiy for each class
	# size = 1 * # classes
	def __calc_class_prob(self):
		log_prob_by_class = []

		for (classif, vectors) in self.ClassifDict.iteritems():
			log_prob_by_class.append(np.log(len(vectors) / float(len(self.train_X))))

		return log_prob_by_class

	# calculate P(x|y)
	# returns a list of lists
	# size = # classes * # features
	# Uses Laplace smoothing
	def __calc_conditional_prob(self):

		#CHECK THE LAPLACE SMOOTHING IS CORRECT
		#adding laplace smoothing parameter... if no example from that class,
		#it reduces to a prior probability of Pr=1/(|features|)
		num_features = len(self.train_X[0])
		count = len(self.ClassifDict.keys()) + 2

		feature_counts = []

		#laplace smoothing
		for (classif, vect_list) in self.ClassifDict.iteritems():
			new_vect = [sum(x) + self.smoothingParam for x in izip(*vect_list)]
			feature_counts.append(new_vect)

		sums_of_features = [sum(f) for f in feature_counts]
	
		feat_log_prob_by_class = []
		for i  in range(len(feature_counts)):
			log_prob = []
			for feat in feature_counts[i]:
				log_prob.append(np.log(feat / sums_of_features[i]))
			feat_log_prob_by_class.append(log_prob)

		return feat_log_prob_by_class

	
	#calculate P(y|x)
	#returns a list of lists
	# size = len(X) * # classes
	def __calc_log_prob(self, X):

		x_log_prob_by_class = []
		x_feature_prob_by_class = []

		feat_log_prob_by_class = self.__calc_conditional_prob()
	
		for x in X:
			#using numpy array to make matrix multiplication easier
			#multipling each feature's log probability into x
			x_feature_prob_by_class.append(np.array(feat_log_prob_by_class) * x)

		
		for feat in x_feature_prob_by_class:
			#sum [ log P(y_i) + sum [log P(x_i,j | y_i)]]
			#add up the probabilities of all the features in a class and add the class's log probability
			x_log_prob_by_class.append((self.log_prob_by_class + feat.sum(axis=1)).tolist())

		return x_log_prob_by_class
			

	#returns the 
	def predict(self, X):
		max = np.argmax(self.__calc_log_prob(X), axis=1)
		return max

	def getValidationAccuracy(self, predictions, validationSet=None):

		if not validationSet: validationSet = self.train_Y #1-fold validation

		correct = 0
		for y in range(len(validationSet)):
			if(validationSet[y] == predictions[y]):
				correct += 1
		return (float(correct) / len(validationSet)) * 100.0



# nb = MultinomialNaiveBayes([[2,1,0,0,0,0], [2,0,1,0,0,0], [1,0,0,1,0,0], [1,0,0,0,1,1]], [1,2,3,4], [])
# print nb.predict([[3,0,3,0,0,1], [2,2,1,0,1,4]])
