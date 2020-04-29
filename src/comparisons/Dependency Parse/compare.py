import sys, os

folds = 3

def extractFromGoldLabels(filePath):

	f = open(filePath, 'r')

	multiWordAT = []

	for line in f:
		aspectTerms = line.split('#')[-1][:-1].split('&')

		res = []

		for aspectTerm in aspectTerms:
			if ' ' in aspectTerm:
				res.append(aspectTerm)

		multiWordAT.append(res)

	f.close()

	return multiWordAT

def extractFromCRF(filePath):
	f = open(filePath, 'r')

	multiWordAT = []
	res = []
	word = ''

	for line in f:

		if line == '\n':
			multiWordAT.append(res)
			res = []

		else:
			line = line[:-1]
			line = line.split('\t')
			if line[-1] == 'TRUE':
				if len(word) > 0:
					word += ' ' + line[1]
				else:
					word = line[1]

			elif line[-1] == 'FALSE':
				if len(word) > 0 and ' ' in word:
					res.append(word)
					
				word = ''

	return multiWordAT

total = 0
newFound = 0
oldDeleted = 0

# run for each fold
for i in range(folds):
	goldLabelFile = '../goldLabel/aspectDev' + str(i) + '.txt'
	allFeaturesFile = '../allFeatures/crf++_predEng' + str(i) + '.txt'
	withoutDPFile = './withoutDP/crf++_predEng' + str(i) + '.txt'

	goldLabels = extractFromGoldLabels(goldLabelFile)
	allFeatures = extractFromCRF(allFeaturesFile)
	withoutDP = extractFromCRF(withoutDPFile)

	for i in range(len(goldLabels)):
		total += len(goldLabels[i])
		for j in range(len(allFeatures[i])):
			if allFeatures[i][j] in goldLabels[i] and allFeatures[i][j] not in withoutDP[i]:
				newFound += 1

		for j in range(len(withoutDP[i])):
			if withoutDP[i][j] in goldLabels[i] and withoutDP[i][j] not in allFeatures[i]:
				oldDeleted += 1

print(total, newFound, oldDeleted)

