import os

filePaths = ['SVMAllFeatureOutput.txt', 'SVMWithoutSOOutput.txt',  'SVMWithoutSentOutput.txt', 'SVMWithoutMCLOutput.txt']

wordSets = []

for i in range(len(filePaths)):
	wordSets.append(set())

	f = open(filePaths[i], 'r')

	for line in f:
		wordSets[-1].add(line.split('\t')[0])

res = []

f = open('SentEvalOutput.txt', 'w')

for i in range(1, len(wordSets)):
	res.append(
		{
			'NewFound': len(wordSets[0] - wordSets[i]),
			'OldLost': len(wordSets[i] - wordSets[0])
		}
	)
	x = list(wordSets[0] - wordSets[i])
	f.write(filePaths[i] + '\n')
	for j in range(len(x)):
		f.write(x[j] + '\n')

for i in range(1, len(filePaths)):
	print(filePaths[i], res[i-1])



