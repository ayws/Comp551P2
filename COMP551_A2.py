from preprocess import *
import random
from naive_bayes import *
import csv
from sklearn.model_selection import KFold, cross_val_score


if __name__ == "__main__":


	######################## VALIDATE NAIVE BAYES #######################
	
	train_XFile = './data/train_set_x.csv'
	processed_X, vect = preprocess(train_XFile)
	processed_Y = preprocessYVals('./data/train_set_y.csv')


	accuracies = []

	valAccFile = open('accuracies_tfidf2.csv', 'w')
	fieldnames = ['Accuracy']
	fileWriter = csv.DictWriter(valAccFile, fieldnames=fieldnames)
	fileWriter.writeheader()

	kf = KFold(n_splits=3, shuffle=True)
	accuracies = []
	processed_X = np.array(processed_X)
	processed_Y = np.array(processed_Y)
	for train, test in kf.split(processed_X):
		train_set_X, test_set_X = processed_X[train], processed_X[test]
		train_set_Y, test_set_Y = processed_Y[train], processed_Y[test]

		nb = MultinomialNaiveBayes(train_set_X, train_set_Y)
		predictions = nb.predict(test_set_X)
		accuracy = nb.getValidationAccuracy(predictions, test_set_Y)
		accuracies.append(accuracy)
		fileWriter.writerow({'Accuracy': accuracy})
		print 'ACCURACY OF CURRENT FOLD IS:', accuracy

	meanAccuracy = sum(accuracies) / float(len(accuracies))
	print 'MEAN ACCURACY OF FOLDS IS:', meanAccuracy


	############################## ACTUAL PREDICTIONS ######################
	train_set_X = processed_X
	train_set_Y = processed_Y
	test_set = preprocess('./data/test_set_x.csv', testVect=vect)
	final_nb = MultinomialNaiveBayes(train_set_X, train_set_Y)
	predictions = final_nb.predict(test_set)

	with open('predictions_tfidf2.csv', 'w') as predictFile:
		fieldnames = ['Id', 'Category']
		writer = csv.DictWriter(predictFile, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(predictions)):
			writer.writerow({'Id': i, 'Category': predictions[i]})

	