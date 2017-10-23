import codecs
import csv

def toList(filename):
	data = csv.reader(codecs.open(filename, 'r'))
	data_list = []
	for row in data:
		if row[1]: data_list.append(row[1])
		else: data_list.append("")
	del data_list[0]
	return data_list


def countOccurances(data):

	count_0 = 0
	count_1 = 0
	count_2 = 0
	count_3 = 0
	count_4 = 0

	for x in data:
		if x[0] == '0': count_0 += 1
		elif x[0] == '1': count_1 += 1
		elif x[0] == '2': count_2 += 1
		elif x[0] == '3': count_3 += 1
		elif x[0] == '4': count_4 += 1

	print 'classified as 0:', count_0
	print 'classified as 1:', count_1
	print 'classified as 2:', count_2
	print 'classified as 3:', count_3
	print 'classified as 4:', count_4

def compare(f1, f2):

	data = zip(toList(f1), toList(f2))
	total = len(data)
	same = 0
	

	for x in data:
		if (x[0] == x[1]): same += 1
	print (float(same) / total * 100)

print "COMPARING SKLEARN'S MODEL WITH ALPHA=1.0 AND OUR IMPLEMENTATION"
compare('./predictions_tfidf.csv', './predictions_sklearn2.csv')
print 'COMPARING COUNT VECTORS AND RANDOM BASELINE'
compare('./predictions_count_vector.csv', './random_baseline.csv')
print 'COMPARING TF-IDF UNIGRAM & BIGRAM VECTORS AND RANDOM BASELINE'
compare('./predictions_tfidf.csv', './random_baseline.csv')
print 'COMPARING TF-IDF UNIGRAM VECTORS AND RANDOM BASELINE'
compare('./predictions_tfidf_unigrams.csv', './random_baseline.csv')
print 'COMPARING TF-IDF UNIGRAM VECTORS AND TF-IDF UNIGRAM & BIGRAM VECTORS'
compare('./predictions_tfidf_unigrams.csv', './predictions_tfidf.csv')
print '_______________________'
print 'CLASSFICATION BREAKDOWNS FOR BASELINE'
countOccurances(toList('./random_baseline.csv'))
print 'CLASSFICATION BREAKDOWNS FOR COUNT VECTORS'
countOccurances(toList('./predictions_count_vector.csv'))
print 'CLASSFICATION BREAKDOWNS FOR TF-IDF UNIGRAM VECTORS'
countOccurances(toList('./predictions_tfidf_unigrams.csv'))
print 'CLASSFICATION BREAKDOWNS FOR TF-IDF UNIGRAM & BIGRAM VECTORS'
countOccurances(toList('./predictions_tfidf.csv'))