import sys
import numpy as np

outPath = '../output/'
dataPath = '../data/'
parseFile = outPath+'parse.output.txt'
dataFile = dataPath+'aspectTerm.txt'
trainFile = outPath+'parseProcessedTrain.txt'
testFile = outPath+'parseProcessedTest.txt'
outFile = outPath+'parseProcessed.txt'
trainPoint = outPath+'aspectTrain.txt'
testPoint = outPath+'aspectTest.txt'

# Function for reading a file
def readFile(filPath):
    fil = open(filPath)
    vec = fil.readlines()
    fil.close()
    return vec

# Extracts data from Parse File 
def extractData():
    vecDataFull = readFile(dataFile)
    vecData = [ele.strip().split('#')[-1] for ele in vecDataFull]

    vecParse = readFile(parseFile)
    
    vecParseElems = [ele.strip().split('\t') for ele in vecParse]

    return vecDataFull,vecData,vecParse,vecParseElems

def addLabels(vecData,vecParse,vecParseElems):
    parseDict = {}
    parseIdx = 0
    for idx,termList in enumerate(vecData):
        allTerms = []
        if termList != '':
            for aspectTerm in termList.split('&'):
                curTerms = aspectTerm.split(' ')
                allTerms += curTerms
        parseDict[idx] = parseIdx

        while (vecParse[parseIdx] != '\n'):
            isAspect = False 
            for term in allTerms:
                if term in vecParseElems[parseIdx][1]:
                    isAspect = True
                    break
            if isAspect: 
                vecParseElems[parseIdx].append('TRUE')
            else:
                vecParseElems[parseIdx].append('FALSE')
            parseIdx += 1
        parseIdx += 1
    
    return parseDict,vecParseElems


# Cleans dependency parse
def cleanParse(vecParse,vecParseElems):
    for parseIdx in range(len(vecParse)):
        if vecParse[parseIdx] == '\n':
            continue
        if vecParseElems[parseIdx][3] == ':':
            continue
        vecParseElems[parseIdx][3] = vecParseElems[parseIdx][3].split(':')[0] 
    return vecParseElems

# Generates Training and Testing data files
def genFiles(vecData,parseDict,vecDataFull,vecParseElems):
    np.random.seed(1)
    arr = np.arange(len(vecData))
    np.random.shuffle(arr)

    trainIdx = int(0.8*len(vecData))

    fil1 = open(trainFile,'w')
    fil2 = open(trainPoint,'w')

    for idx in arr[:trainIdx]:
        parseIdx = parseDict[idx]
        fil2.write(vecDataFull[idx])

        while len(vecParseElems[parseIdx]) != 1:
            if len(vecParseElems[parseIdx]) != 7:
                print(vecParseElems[parseIdx])
            fil1.write('\t'.join(vecParseElems[parseIdx]))
            fil1.write('\n')
            parseIdx += 1
        fil1.write('\n')

    fil3 = open(testFile,'w')
    fil4 = open(testPoint,'w')

    for idx in arr[trainIdx:]:
        parseIdx = parseDict[idx]
        
        vecTestData = vecDataFull[idx].strip().split('#')
        finalvec = vecTestData[:-2]+[vecTestData[-1]]
        fil4.write('#'.join(finalvec))
        fil4.write('\n')

        while len(vecParseElems[parseIdx]) != 1:
            fil3.write('\t'.join(vecParseElems[parseIdx][:-1]))
            fil3.write('\n')
            parseIdx += 1
        fil3.write('\n')

if __name__ == '__main__':
    vecDataFull,vecData,vecParse,vecParseElems = extractData()
    parseDict,vecParseElems = addLabels(vecData,vecParse,vecParseElems)
    vecParseElems = cleanParse(vecParse,vecParseElems)
    genFiles(vecData,parseDict,vecDataFull,vecParseElems)


# Generates feature to indicate whether a word is a Hindi or an English word.
def engFeatures():
    vec1 = readFile(trainPath)
    vec2 = readFile('engwords.txt')
    
    vec2 = set([ele.strip() for ele in vec2])
    vec3 = [ele.strip().split("\t") for ele in vec1]
    vec4 = [ele+[ele[-1]] for ele in vec3]
    for idx,ele in enumerate(vec4):
        if ele[1] in vec2:
            print(idx)
            ele[-2] = 'TRUE'
        else:
            ele[-2] = 'FALSE'
    fil3 = open("eng_features.txt","w")
    for ele in vec4:
        fil3.write(" ".join(ele)+'\n')
    fil3.close()




