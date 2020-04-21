import sys
import numpy as np

outPath = '../output/'
dataPath = '../data/'
parseFile = outPath+'parse.output.txt'
dataFile = dataPath+'aspectTerm.txt'
outFile = outPath+'parseProcessed.txt'

devPoint = outPath+'aspectDev'
# devPoint2 = outPath+'aspectDev2.txt'
# devPoint3 = outPath+'aspectDev3.txt'

trainPoint = outPath+'aspectTrain.txt'
testPoint = outPath+'aspectTest.txt'

devFileTrain = outPath+'parseProcessedTrain'
# devFileTrain23 = outPath+'parseProcessedTrain23.txt'
# devFileTrain31 = outPath+'parseProcessedTrain31.txt'

devFileTest = outPath+'parseProcessedTest'
# devFileTest2 = outPath+'parseProcessedTest2.txt'
# devFileTest3 = outPath+'parseProcessedTest3.txt'

trainFile = outPath+'parseProcessedTrain.txt'
testFile = outPath+'parseProcessedTest.txt'

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

def writeData(arr,startIdx,endIdx,filPath1,test=False):
    fil1 = open(filPath1,'w')
    if startIdx>endIdx:
        curArr = np.concatenate((arr[startIdx:],arr[:endIdx]))
    else:
        curArr = arr[startIdx:endIdx]

    for idx in curArr:
        parseIdx = parseDict[idx]

        while len(vecParseElems[parseIdx]) != 1:
            if len(vecParseElems[parseIdx]) != 7:
                print(vecParseElems[parseIdx])
            if test:
                fil1.write('\t'.join(vecParseElems[parseIdx][:-1]))
            else:
                fil1.write('\t'.join(vecParseElems[parseIdx]))
            fil1.write('\n')
            parseIdx += 1
        fil1.write('\n')
    fil1.close()

def writePoint(arr,startIdx,endIdx,filPath2):
    fil2 = open(filPath2,'w')
    for idx in arr[startIdx:endIdx]:
        fil2.write(vecDataFull[idx])
    fil2.close()

# Generates Training and Testing data files
def genFiles(vecData,parseDict,vecDataFull,vecParseElems):
    np.random.seed(1)
    arr = np.arange(len(vecData))
    np.random.shuffle(arr)

    trainIdx = int(0.8*len(vecData))
    devIdx1 = int(len(vecData)/3)
    devIdx2 = int(2*len(vecData)/3)

    # fil1 = open(trainFile,'w')
    # fil2 = open(trainPoint,'w')

    # fil3 = open(testFile,'w')
    # fil4 = open(testPoint,'w')

    # fil5 = open(devFile1,'w')
    # fil6 = open(devFile1,'w')
    # fil7 = open(devFile3,'w')

    # fil8 = open(devPoint1,'w')
    # fil9 = open(devPoint2,'w')
    # fil10 = open(devPoint3,'w')

    writeData(arr,0,trainIdx,trainFile)
    writePoint(arr,0,trainIdx,trainPoint)

    writePoint(arr,trainIdx,len(vecData),testPoint)
    writeData(arr,trainIdx,len(vecData),testFile,True)

    indices = [0,devIdx1,devIdx2,len(vecData)]

    for ele in range(3):
        if ele<2:
            writeData(arr,indices[ele],indices[(ele+2)],devFileTrain+str(ele)+str((ele+1)%3)+".txt")
        else:
            writeData(arr,indices[ele],indices[1],devFileTrain+str(ele)+str((ele+1)%3)+".txt")
        writeData(arr,indices[ele],indices[ele+1],devFileTest+str(ele)+'.txt',True)
        writePoint(arr,indices[ele],indices[ele+1],devPoint+str(ele)+'.txt')


    # writeData(arr,0,devIdx2,devFileTrain12)
    # writeData(arr,devIdx1,len(vecData),devFileTrain23)
    # writeData(arr,devIdx2,devIdx1,devFileTrain31)   

    # writeData(arr,0,devIdx1,devFileTest1,True)
    # writeData(arr,devIdx1,devIdx2,devFileTest2,True)  
    # writeData(arr,devIdx2,len(vecData),devFileTest3,True)
    
    
    
    # writePoint(arr,0,devIdx1,devPoint1)
    # writePoint(arr,devIdx1,devIdx2,devPoint2)
    # writePoint(arr,devIdx2,len(vecData),devPoint3)

    

    



    

    # for idx in arr[trainIdx:]:
    #     parseIdx = parseDict[idx]
        
    #     vecTestData = vecDataFull[idx].strip().split('#')
    #     finalvec = vecTestData[:-2]+[vecTestData[-1]]
    #     fil4.write('#'.join(finalvec))
    #     fil4.write('\n')

    #     while len(vecParseElems[parseIdx]) != 1:
    #         fil3.write('\t'.join(vecParseElems[parseIdx][:-1]))
    #         fil3.write('\n')
    #         parseIdx += 1
    #     fil3.write('\n')

# Generates feature to indicate whether a word is a Hindi or an English word.
def engFeatures(featurePath,engFeaturePath,hindiWordsPath,train=True):
    wordSet = set([])
    aspectSet = set([])

    vec1 = readFile(featurePath)
    vec2 = readFile(hindiWordsPath)
    
    vec2 = set([ele.strip() for ele in vec2])
    vec3 = [ele.strip().split("\t") for ele in vec1]
    vec4 = [ele+[ele[-1]] for ele in vec3]
    # import ipdb; ipdb.set_trace()
    for idx,ele in enumerate(vec4):
        if ele[0] == '':
            continue
        if ((ele[1] in vec2) or (ele[2] in vec2)):
            # print(idx)
            if train:
                ele[-2] = 'FALSE'
                if ele[-1] == 'TRUE':
                    aspectSet.add(ele[1])
            else:
                ele[-1] = 'FALSE'
        else:
            if train:
                ele[-2] = 'TRUE'
                wordSet.add(ele[1])
            else:
                ele[-1] = 'TRUE'
                wordSet.add(ele[1])
    fil3 = open(engFeaturePath,"w")
    for ele in vec4:
        fil3.write(" ".join(ele)+'\n')
    fil3.close()
    if train:
        return wordSet,aspectSet
    else:
        return wordSet

if __name__ == '__main__':
    vecDataFull,vecData,vecParse,vecParseElems = extractData()
    parseDict,vecParseElems = addLabels(vecData,vecParse,vecParseElems)
    vecParseElems = cleanParse(vecParse,vecParseElems)
    genFiles(vecData,parseDict,vecDataFull,vecParseElems)

    # trainWordSet,aspectSet = engFeatures(devFileTrain+'.txt',devFileTrain+'Eng.txt','../output/dependencyTreeBankDictionary.txt',True)
    # testWordSet = engFeatures(devFileTest+'.txt',devFileTest+'Eng.txt','../output/dependencyTreeBankDictionary.txt',False)
    # totalSet = trainWordSet.union(testWordSet)
    # f = open('../output/EngWordFoundSWN.txt','w')
    # f2 = open('../output/HindiAspect.txt','w')
    # for ele in totalSet:
    #     f.write(ele+'\n')
    # for ele in aspectSet:
    #     f2.write(ele+'\n')






