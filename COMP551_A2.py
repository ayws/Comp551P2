from knn import *
from preprocess import *
import random
from naive_bayes import *


if __name__ == "__main__":

	train_X = './data/train_set_x.csv'
	train_Y = toList('./data/train_set_y.csv')
	test_X = './data/test_set_x.csv'
	# randBaseline = toList('./data/random_baseline.csv')


	#ADD IN A VALIDATION SET

	train_set_X = preprocess(train_X, False)
	test_set_X = preprocess(test_X, True)

	rand = random.random()
	random.shuffle(train_set_X)
	random.shuffle(train_Y)

	trial_X = train_set_X[:1000]
	trial_Y = train_Y[:1000]
	test_X = train_set_X[1001:2001]
	test_Y = train_Y[1001:2001]

	# knnClassifier = KthNearestNeighbour(trial_X, trial_Y, test_X)


	#choosing k = N^(1/2)
	# k = int(pow(len(trial_X), 0.5))
	# print 'chose value k =', k
	# predictions = knnClassifier.predict(k)

	# for x in range(len(predictions)):
	# 	print 'predicted=', predictions[x]
	# print knnClassifier.getValidationAccuracy(predictions, test_Y)

	nb = MultinomialNaiveBayes(trial_X, trial_Y, test_X)
	predictions = nb.predict(test_X)
	print 'PREDICTIONS:', predictions
	print test_Y
	print 'VALIDATION ACCURACY:', nb.getValidationAccuracy(predictions, test_Y)