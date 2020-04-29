# Imports
import sys
import pickle
import io

# Data Paths
trainFile = '../output/aspectTrain.txt'
testFile = '../output/aspectTest.txt'
tempFile = 'temp.txt'
sentimentFile = 'HSWN_WN.txt'

# Hyperparameters
windowSize = 10

# Function for reading a file
def readFile(filPath):
	fil = open(filPath, 'r')
	vec = fil.readlines()
	fil.close()
	return vec

# Function for opening PK file
def openPKFile(path):
	with open(path, 'rb') as f:
		pkFile = pickle.load(f)
	f.close()
	return pkFile

# Hindi WordNet files
WordSynsetDict = openPKFile('../HindiWordNet/WordSynsetDict.pk')
SynsetOnto = openPKFile('../HindiWordNet/SynsetOnto.pk')
SynsetWords = openPKFile('../HindiWordNet/SynsetWords.pk')

wordsPolarities = {}

# Function for extracting window of text around an aspect term
def extractWindow(aspectTerm, wordList):

	fillerWord = '*'

	aspectTermWords = aspectTerm.split(' ')
	aspectSize = len(aspectTermWords)
	remWindowSize = windowSize-aspectSize

	startIndex = 0
	for idx, word in enumerate(wordList):
		if aspectTermWords[0] in word:
			startIndex = idx
			break
	window = []
	windowStart = max(0, startIndex - int(remWindowSize/2))
	emptyPlacesStart = -1 * (startIndex - int(remWindowSize/2))

	# add filler word at the start
	if emptyPlacesStart > 0:
		window += [fillerWord] * emptyPlacesStart

	windowEnd = min(startIndex+aspectSize+int(remWindowSize/2), len(wordList))
	window += wordList[windowStart:windowEnd]

	emptyPlacesEnd = -1 * len(wordList) - startIndex+aspectSize+int(remWindowSize/2)
	# add filler word at the end
	if emptyPlacesEnd > 0:
		window += [fillerWord] * emptyPlacesEnd

	# if size still not equal to window size add more filler words
	if len(window) < windowSize:
		remainingPlaces = windowSize - len(window)
		if len(window[-1]) == 0 or window[-1][-1] == u'।':
			window = [fillerWord] * remainingPlaces + window
		else:
			window += [fillerWord] * remainingPlaces

	return ' '.join(window)


def generate_lexicon():
	lexicon = {}
	swn_file = "HSWN_WN.txt"
	with io.open(swn_file, 'r', encoding='utf8') as f:
		for line in iter(f):
			line = line.rstrip()
			if line:
				data = line.split()
				pos_score = float(data[2])
				neg_score = float(data[3])
				words = data[4]
				words = words.split(',')
				for word in words:
					word_map = {}
					word_map['positive'] = pos_score
					word_map['negative'] = neg_score
					lexicon[word] = word_map
	return lexicon

def getWordPolarity(word):
	if word in wordsPolarities:
		if wordsPolarities[word]['positive'] > wordsPolarities[word]['negative']:
			return wordsPolarities[word]['positive']
		return -1 * wordsPolarities[word]['negative']
	return 0


# def getSentimentVector():
def runWindowSentimentExtraction():

	wordsPolarities = generate_lexicon()
	print(len(wordsPolarities))

	data = readFile(trainFile)
	for line in data:
		if line[-1] == '\n':
			line = line[:-1]
		line = line.split('#')
		wordList = line[1].split(' ')
		if len(line[4]) > 0:
			aspectTerms = line[4].split('&')
			for aspectTerm in aspectTerms:
				window = extractWindow(aspectTerm, wordList).split(' ')

				result = []

				for word in window:
					if word == '*':
						result.append(0)
					else:
						result.append(getWordPolarity(word))
				if max(result) != 0:
					print(result)
		

runWindowSentimentExtraction()

