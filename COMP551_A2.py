from knn import *
from preprocess import *


if __name__ == "__main__":

	train_X = './data/train_set_x.csv'
	train_Y = toList('./data/train_set_y.csv')
	test_X = './data/test_set_x.csv'
	# randBaseline = toList('./data/random_baseline.csv')


	#ADD IN A VALIDATION SET

	train_set_X = preprocess(train_X, False)
	test_set_X = preprocess(test_X, True)
	knnClassifier = KthNearestNeighbour(train_set_X, train_Y, test_set_X)
	knnClassifier.predict()
	# trainKNN(train_set_X, train_Y, test_set_X)