import os
import sys

featureList = [
	'Ua:%x[-2,1]\nUb:%x[-1,1]\nUc:%x[0,1]\nUd:%x[1,1]\nUe:%x[2,1]',
	'Uf:%x[0,5]',
	'B',
	'Ug:%x[-2,3]\nUh:%x[-1,3]\nUi:%x[0,3]\nUj:%x[1,3]\nUk:%x[2,3]',
	'Um:%x[0,6]',
	'Ul:%x[0,1]/%x[0,3]'
]

tasks = {
	'ATE': {
		'crf': {
			'runTools': 'python3.6 runTools.py CRF True',
			'runCommand': 'python3.6 -W ignore::FutureWarning aspectExtraction.py CRF crf++_predEng',
			'resultFile': '../output/evaluations/crf.txt'
		},
		'memm': {
			'runTools': 'python3.6 runTools.py MEMM True',
			'runCommand': 'python3.6 -W ignore::FutureWarning aspectExtraction.py MEMM memm_pred',
			'resultFile': '../output/evaluations/memm.txt'
		}
	},
	'SA': {}
}

featureFilePath = 'patternFile.txt'

path, dirs, files = next(os.walk("../output/evaluations"))
fileCount = len(files)

evalFilePath = '../output/evaluations/eval' + str(fileCount) + '.txt'

featureVariations = []

for i in range(len(featureList)):
	featureVariations.append([])
	for j in range(i+1):
		featureVariations[-1].append(j)

def featureString(features):

	for i in range(len(features)):
		features[i] = str(features[i])

	return ','.join(features)

evalFile = open(evalFilePath, 'w')

for i in range(len(featureList)):
	evalFile.write('# Feature ' + str(i) + '\n')
	evalFile.write(featureList[i] + '\n')

evalFile.write('\n')

for variation in featureVariations:

	featureFile = open(featureFilePath, 'w')

	for i in range(len(variation)):
		featureFile.write(featureList[variation[i]] + '\n')

	featureFile.close()

	for task in tasks.keys():
		for model in tasks[task].keys():
			os.system(tasks[task][model]['runTools'])
			os.system(tasks[task][model]['runCommand'] + ' > ' + tasks[task][model]['resultFile'])

			resultFile = open(tasks[task][model]['resultFile'], 'r')

			evalFile.write(task + '\t' + model + '\t' + featureString(variation) + '\t' + resultFile.readlines()[-1])

			resultFile.close()

evalFile.close()
