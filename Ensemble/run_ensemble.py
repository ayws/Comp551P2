import sys
import random
import csv
import numpy as np

sys.path.append('..')
from feature_extraction import *
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn.ensemble import BaggingClassifier

if __name__ == "__main__":

	### HELPER FUNCTIONS ###

	def getValidationAccuracy(predictions, validationSet):

		correct = 0
		for y in range(len(validationSet)):
			if(validationSet[y] == predictions[y]):
				correct += 1

		print correct
		return (float(correct) / len(validationSet)) * 100.0

	### VALIDATE ENSEMBLE ###

	train_XFile = '../data/train_set_x.csv'
	processed_X, vect = preprocess(train_XFile)
	processed_Y = preprocessYVals('../data/train_set_y.csv')


	accuracies = []

	#4-Fold cross-validation

	kf = KFold(n_splits=5, shuffle=True)
	accuracies = []
	processed_X = np.array(processed_X)
	processed_Y = np.array(processed_Y)
	for train, test in kf.split(processed_X):
		train_set_X, test_set_X = processed_X[train], processed_X[test]
		train_set_Y, test_set_Y = processed_Y[train], processed_Y[test]

		knn = KNeighborsClassifier(n_neighbors = 2, weights = 'distance')
		bag_knn = BaggingClassifier(knn, max_samples=0.5, max_features=0.5)
		bag_knn.fit(train_set_X, train_set_Y)
		predictions = bag_knn.predict(test_set_X)
		accuracy = getValidationAccuracy(predictions, test_set_Y)
		accuracies.append(accuracy)
		print 'ACCURACY OF CURRENT FOLD IS:', accuracy

	meanAccuracy = sum(accuracies) / float(len(accuracies))
	print 'KNN MEAN ACCURACY OF FOLDS IS:', meanAccuracy

	accuracy = 0

	for train, test in kf.split(processed_X):
		train_set_X, test_set_X = processed_X[train], processed_X[test]
		train_set_Y, test_set_Y = processed_Y[train], processed_Y[test]

		nb = naive_bayes.MultinomialNB()
		bag_nb = BaggingClassifier(nb, max_samples=0.5, max_features=0.5)
		bag_nb.fit(train_set_X, train_set_Y)
		predictions = bag_nb.predict(test_set_X)
		accuracy = getValidationAccuracy(predictions, test_set_Y)
		accuracies.append(accuracy)
		print 'NB ACCURACY OF CURRENT FOLD IS:', accuracy

	meanAccuracy = sum(accuracies) / float(len(accuracies))
	print 'KNN MEAN ACCURACY OF FOLDS IS:', meanAccuracy

	### CREATE PREDICTIONS ###

	train_set_X = processed_X
	train_set_Y = processed_Y
	test_set = preprocess('../data/test_set_x.csv', testVect=vect)

	print 'Creating KNN bag prediction'

	final_bag_knn = bag_knn.fit(train_set_X, train_set_Y)
	predictions = final_bag_knn.predict(test_set)

	with open('predictions_ensemble_knn.csv', 'w') as predictFile:
		fieldnames = ['Id', 'Category']
		writer = csv.DictWriter(predictFile, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(predictions)):
			writer.writerow({'Id': i, 'Category': predictions[i]})

	print 'Creating NB bag prediction'

	final_bag_nb = bag_nb.fit(train_set_X, train_set_Y)
	predictions = final_bag_nb.predict(test_set)

	with open('predictions_ensemble_nb.csv', 'w') as predictFile:
		fieldnames = ['Id', 'Category']
		writer = csv.DictWriter(predictFile, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(predictions)):
			writer.writerow({'Id': i, 'Category': predictions[i]})
