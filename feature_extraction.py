import codecs
import csv
from string import digits
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def toList(filename):
	data = csv.reader(codecs.open(filename, 'r'))
	data_list = []
	for row in data:
		if row[1]:
			data_list.append("".join(row[1].translate(None,digits).split()))
		else:
			data_list.append("")
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

'''
parameters: 
	filename :string, self-explanatory
	testVect: a TfidfVectorizer instance. Only passed if we're preprocessing the test set.
'''
def preprocess(filename, testVect=None):

	data = toList(filename)

	#tokenize each sentence into characters
	#(1,1) = unigrams, (1,2) = unigrams & bigrams
	vect = TfidfVectorizer( analyzer='char',
							strip_accents=None,
							ngram_range=(1,2),
							lowercase=True,
							token_pattern='(?u)\\b\\w+\\b'
							)


	if testVect: 
		matrix = testVect.transform(data).toarray()
		return matrix
	else: 
		matrix = vect.fit_transform(data).toarray()
		return matrix, vect


