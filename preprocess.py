import codecs
import sys
import csv
from string import digits
from nltk import FreqDist, bigrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from operator import itemgetter
from math import log
from collections import Counter



def toList(filename):

	data = csv.reader(codecs.open(filename, 'r'))
	data_list = []
	for row in data:
		if row[1]:
			data_list.append("".join(row[1].translate(None,digits).split()))
		else:
			data_list.append("")
	# data = ["".join(row[1].translate(None, digits).split()) for row in data if row[1]]
	del data_list[0] #remove header
	return data_list

def preprocessYVals(filename):
	data = csv.reader(codecs.open(filename, 'r'))
	data_list = []
	for row in data:
		if row[1]:
			data_list.append(row[1])
		else:
			data_list.append(None)
	del data_list[0] #remove header
	data_list = [int(x) for x in data_list]
	return data_list

#test is a boolean indicating whether or not it's a test set
def preprocess(filename, testVect=None):
	print 'preprocessing set from', filename


	data = toList(filename)

	#tokenize each sentence into characters


	vect = CountVectorizer( analyzer='char',
							strip_accents=None,
							ngram_range=(1,2),
							lowercase=True,
							token_pattern='(?u)\\b\\w+\\b'
							)

	# feature_names = vect.get_feature_names()

	if testVect: 
		matrix = testVect.transform(data).toarray()
		return matrix
	else: 
		matrix = vect.fit_transform(data).toarray()
		return matrix, vect


