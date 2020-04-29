import os, sys
import string

aspectTermFilePath = '../../data/aspectTerm.txt'

aspectTermFile = open(aspectTermFilePath, 'r')

wordSet = set()

for line in aspectTermFile:
	line = line.split('#')[1]

	exclude = set(string.punctuation)
	line = ''.join(ch for ch in line if ch not in exclude)

	if line[-1] == u'।':
		line = line[:-1]

	words = line.split(' ')
	for word in words:
		wordSet.add(word)

print(len(wordSett))

