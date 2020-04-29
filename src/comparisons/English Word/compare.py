import sys, os

folds = 3

# Create a hindi dictionary
hindiDict = set()
hindiDictFilePath = '../../../output/WordSynsetDict.txt'
hindiDictFile = open(hindiDictFilePath, 'r')

for word in hindiDictFile:
	if word[-1] == '\n':
		word = word[:-1]

	hindiDict.add(word)

hindiDictFile.close()



def extractFromGoldLabels(filePath):

	f = open(filePath, 'r')

	englishWordAT = []

	for line in f:
		aspectTerms = line.split('#')[-1][:-1].split('&')

		res = []

		for aspectTerm in aspectTerms:
			words = aspectTerm.split(' ')
			words = list(filter(('').__ne__, words))
			for word in words:
				if word not in hindiDict:
					res.append(aspectTerm)
					break

		englishWordAT.append(res)

	f.close()

	return englishWordAT

def extractFromCRF(filePath):
	f = open(filePath, 'r')

	englishWordAT = []
	res = []
	words = ''

	for line in f:

		if line == '\n':
			englishWordAT.append(res)
			res = []

		else:
			line = line[:-1]
			line = line.split('\t')
			if line[-1] == 'TRUE':
				if len(words) > 0:
					words += ' ' + line[1]
				else:
					words = line[1]

			elif line[-1] == 'FALSE' and len(words) > 0:
				words = words.split(' ')
				words = list(filter(('').__ne__, words))
				for word in words:
					if word not in hindiDict:
						res.append(word)
						break
					
				words = ''

	return englishWordAT

total = 0
newFound = 0
oldDeleted = 0

# temp = extractFromGoldLabels('../goldLabel/aspectDev0.txt')

# print(len(temp))

# f = open('temp.txt', 'w')

# for i in range(len(temp)):
# 	for j in range(len(temp[i])):
# 		f.write(temp[i][j] + '\n')

# run for each fold
for i in range(folds):
	goldLabelFile = '../goldLabel/aspectDev' + str(i) + '.txt'
	allFeaturesFile = '../allFeatures/crf++_predEng' + str(i) + '.txt'
	withoutEnglishWordsFile = './withoutEnglishWords/crf++_predEng' + str(i) + '.txt'

	goldLabels = extractFromGoldLabels(goldLabelFile)
	allFeatures = extractFromCRF(allFeaturesFile)
	withoutEnglishWords = extractFromCRF(withoutEnglishWordsFile)

	# print(len(goldLabels), len(allFeatures), len(withoutEnglishWords))

	for i in range(len(goldLabels)):
		total += len(goldLabels[i])
		for j in range(len(withoutEnglishWords[i])):
			if withoutEnglishWords[i][j] in goldLabels[i] and withoutEnglishWords[i][j] not in allFeatures[i]:
				newFound += 1
				print('without', withoutEnglishWords[i][j])

		for j in range(len(allFeatures[i])):
			if allFeatures[i][j] in goldLabels[i] and allFeatures[i][j] not in withoutEnglishWords[i]:
				oldDeleted += 1
				print(allFeatures[i][j])

print(total, newFound, oldDeleted)

