from knn import *
from preprocess import *
import random
from naive_bayes import *
import csv
from sklearn.model_selection import KFold, cross_val_score


# def k_fold_split(dataset_X, dataset_Y, k):


if __name__ == "__main__":





	################# KNN CLASSIFIER STUFF ######################

	# knnClassifier = KthNearestNeighbour(trial_X, trial_Y, test_X)

	#choosing k = N^(1/2)
	# k = int(pow(len(trial_X), 0.5))
	# print 'chose value k =', k
	# predictions = knnClassifier.predict(k)

	# for x in range(len(predictions)):
	# 	print 'predicted=', predictions[x]
	# print knnClassifier.getValidationAccuracy(predictions, test_Y)


	######################## TEST NAIVE BAYES #######################
	
	train_XFile = './data/train_set_x.csv'
	processed_X, vect = preprocess(train_XFile)
	processed_Y = preprocessYVals('./data/train_set_y.csv')


	# train_set_X = processed_X[1000:2000]
	# test_set_X = processed_X[:1000]
	# train_set_Y = processed_Y[1000:2000]
	# test_set_Y = processed_Y[:1000]

	folds = 10
	fold_size = len(processed_X) / folds
	accuracies = []

	# for i in range(folds):
	# 	train_set_X = processed_X[i*fold_size:][:fold_size]
	# 	train_set_Y  = processed_Y[i*fold_size:][:fold_size]
	# 	test_set_X = processed_X[:i*fold_size] + processed_X[(i+1)*fold_size:]
	# 	test_set_Y = processed_Y[:i*fold_size] + processed_X[(i+1)*fold_size:]
	# 	nb = MultinomialNaiveBayes(train_set_X, train_set_Y)
	# 	predictions = nb.predict(test_set_X)
	# 	accuracy = nb.getValidationAccuracy(predictions, test_set_Y)
	# 	accuracies.append(accuracy)
	# 	print 'ACCURACY OF FOLD', i, 'IS:', accuracy

	# meanAccuracy = sum(accuracies) / float(len(accuracies))
	# print 'MEAN ACCURACY OF', folds, 'FOLDS IS:', meanAccuracy

	kf = KFold(n_splits=10, shuffle=True)
	accuracies = []
	processed_X = np.array(processed_X)
	processed_Y = np.array(processed_Y)
	for train, test in kf.split(processed_X):
		print train, test
		train_set_X, test_set_X = processed_X[train], processed_X[test]
		train_set_Y, test_set_Y = processed_Y[train], processed_Y[test]

		nb = MultinomialNaiveBayes(train_set_X, train_set_Y)
		predictions = nb.predict(test_set_X)
		accuracy = nb.getValidationAccuracy(predictions, test_set_Y)
		accuracies.append(accuracy)
		print 'ACCURACY OF CURRENT FOLD IS:', accuracy

	meanAccuracy = sum(accuracies) / float(len(accuracies))
	print 'MEAN ACCURACY OF', folds, 'FOLDS IS:', meanAccuracy



	# nb = MultinomialNaiveBayes(train_set_X, train_set_Y, test_set_X)


	# predictions = nb.predict(test_set_X)
	# accuracy = nb.getValidationAccuracy(predictions, test_set_Y)
	# print 'VALIDATION ACCURACY:', accuracy


	############################## ACTUAL PREDICTIONS ######################
	# train_X = './data/train_set_x.csv'
	# train_Y = preprocessYVals('./data/train_set_y.csv')

	# test_X = './data/test_set_x.csv'
	# train_set_X, vect = preprocess(train_X)		for y in range(len(validationSet)):
	# test_set_X = preprocess(test_X, vect)
	# randBaseline = toList('./data/random_baseline.csv')

	# nb = MultinomialNaiveBayes(train_set_X[:1000], train_Y[:1000], test_set_X[:300])
	# predictions = nb.predict(test_set_X)

	# with open('predictions.csv', 'w') as predictFile:
	# 	fieldnames = ['id', 'classification']
	# 	writer = csv.DictWriter(predictFile, fieldnames=fieldnames)
	# 	writer.writeheader()
	# 	for i in range(len(predictions)):
	# 		writer.writerow({'id': i, 'classification': predictions[i]})

	