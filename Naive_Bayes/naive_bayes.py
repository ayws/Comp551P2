import numpy as np 
from math import sqrt, exp, pow
from operator import add
from itertools import izip

class MultinomialNaiveBayes:

	def __init__(self, train_X, train_Y, smoothingParam=1.0):
		
		self.train_X = train_X
		self.train_Y = train_Y
		self.smoothingParam = smoothingParam
		self.ClassifDict = self.__groupByClass()
		self.class_prior = self.__calc_prior()

		if len(train_X) != len(train_Y):
			print 'Error -- Training sets are not of same length.'
			exit()


	'''
		Returns a dictionary with classes as keys.
		Values are lists of data with the associated key class.
	'''
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

	'''
		Returns a list with P(y), the prior log probability of each class.
		Size: 1 * # classes
	'''
	def __calc_prior(self):

		class_prior = []

		for (classif, vectors) in self.ClassifDict.iteritems():
			class_prior.append(np.log(len(vectors) / float(len(self.train_X))))

		return class_prior

	'''
		Calculates P(x|c) with Laplace Smoothing: P(x|c) = count(x,c)+ delta / (# classes + # features in class c)
		Returns a list of lists of size:  # classes * # features
	'''
	def __calc_likelihood(self):

		num_features = len(self.train_X[0])
		
		feature_counts = []
		for (classif, vect_list) in self.ClassifDict.iteritems():
			new_vect = [sum(x) + self.smoothingParam for x in izip(*vect_list)] #count(x,c)+1 
			feature_counts.append(new_vect)

		sums_of_features = [sum(f) for f in feature_counts]
	
		feat_likelihood = []
		for i  in range(len(feature_counts)):
			log_prob = []
			for feat in feature_counts[i]:
				log_prob.append(np.log(feat / sums_of_features[i]))
			feat_likelihood.append(log_prob)

		return feat_likelihood

	
	'''
		Calculates P(y|x)
		Returns a list of lists of size len(X) * # classes
	'''
	def __calc_posterior(self, X):

		x_posterior_prob = []
		x_feat_prob_by_class = []
		feat_likelihood = self.__calc_likelihood()

		#multiplies each feature's log probability into x using numpy matrix multiplication
		for x in X:
			x_feat_prob_by_class.append(np.array(feat_likelihood) * np.array(x))
		
		#sum [ log P(y_i) + sum [log P(x_i,j | y_i)]]
		#add up the probabilities of all the features in a class and the class's log probability
		for feat in x_feat_prob_by_class:
			x_posterior_prob.append((self.class_prior + feat.sum(axis=1)).tolist())

		return x_posterior_prob
			

	'''
		Returns the max calculated probability row of the numpy array.
	'''
	def predict(self, X):
		return np.argmax(self.__calc_posterior(X), axis=1)

	'''
		Input is two lists of either integers or strings (classes). 
		Used to check accuracy of a fold in cross-validation.
	'''
	def getValidationAccuracy(self, predictions, validationSet):

		correct = 0
		for y in range(len(validationSet)):
			if(validationSet[y] == predictions[y]):
				correct += 1

		print correct
		return (float(correct) / len(validationSet)) * 100.0

