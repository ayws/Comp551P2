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

	print 'BREAKDOWN BY CLASS FOR FILE', f1
	countOccurances(toList(f1))
	print '_________________'
	print 'BREAKDOWN BY CLASS FOR FILE', f2
	countOccurances(toList(f2))


compare('./predictions.csv', './predictions_tfidf.csv')
compare('./predictions_tfidf.csv', './predictions_tfidf2.csv')
