import sys

sentiFilePath = 'SentiWordNet_3.0.0.txt'
translationFilePath = 'translation.txt'

wordScore = {}

sentiFile = open(sentiFilePath, 'r')

for line in sentiFile:
    line = line.split('\t')
    if len(line) > 4 and len(line[0]) > 0 and line[0][0] != '#':
        
        words = line[4]
        PosScore = line[2]
        NegScore = line[3]

        words = words.split(' ')

        for word in words:
            word = word.split('#')[0]
            if word not in wordScore:
                wordScore[word] = [PosScore, NegScore]
            else:
                wordScore[word] = [max(PosScore, wordScore[word][0]), max(NegScore, wordScore[word][1])]
                

sentiFile.close()


translationFile = open(translationFilePath, 'r')
scoreFile = open('scoreFile.txt', 'w')

for line in translationFile:
    if line[-1] == '\n':
        line = line[:-1]
    line = line.split(' ')
    engWord = line[1].lower()
    if line[1] in wordScore:
        line.append(wordScore[engWord][0])
        line.append(wordScore[engWord][1])

    else:
        line.append('0')
        line.append('0')

    scoreFile.write(' '.join(line) + '\n')

translationFile.close()
scoreFile.close()

    





