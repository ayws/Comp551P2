import codecs
import sys
import csv
from nltk import FreqDist, bigrams
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
from math import log
from collections import Counter



def toList(filename):

	data = csv.reader(codecs.open(filename, 'r'))
	data = [row[1] for row in data if row[1]]
	del data[0] #remove header
	return data

#test is a boolean indicating whether or not it's a test set
def preprocess(filename, test):
	print 'preprocessing sets'


	data = toList(filename)

	#tokenize each sentence into characters


	vect = CountVectorizer( analyzer='char',
							ngram_range=(1,2),
							lowercase=True,
							# smooth_idf=True, #smooth weights by adding one to doc freq
							# use_idf=True
							)

	
	# if test: matrix = vect.transform(data).toarray()
	# else: matrix = vect.fit_transform(data).toarray()
	matrix = vect.fit_transform(data).toarray()
	feature_names = vect.get_feature_names()

	return matrix 

