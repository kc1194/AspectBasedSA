import nltk
from nltk.tag import hmm
import io
import pdb
import sys

# Data paths
outPath = "../output/"
dataPath = "../data/"
trainPath = outPath+"parseProcessedTrain.txt"
testPath = outPath+"parseProcessedTest.txt"
testIdPath = outPath+"aspectTest.txt"
goldPath = dataPath+"aspectTerm.txt"

devIdPath = outPath+"aspectDev"
devTrainPath = outPath+"parseProcessedTrain"
devTestPath = outPath+"parseProcessedTest"

# Function for reading a file
def readFile(filPath):
    fil = open(filPath)
    vec = fil.readlines()
    fil.close()
    return vec


# Function for extracting word and label from dependency parse. No labels are returned when testing
def readData(dataPath,train=True):
    vec = readFile(dataPath)
    data = []
    point = [] 

    for ele in vec:
        if (ele=='\n'):
            data.append(point)
            point = []
            continue
        curPoint = ele.strip().split("\t")
        if train:
            point.append((curPoint[1],curPoint[-1]))
        else:
            point.append(curPoint[1])

    return data


# Helper function to read output of maxEnt classifier in wapiti 
def readMax(dataPath):
    vec = readFile(dataPath)
    data = []
    point = [] 
    curIdx = 1

    for ele in vec:
        if ele=='\n':
            continue
        curPoint = ele.strip().split("\t")
        if int(curPoint[0])<curIdx:
            data.append(point)
            point = []
            curIdx = 1
        else:
            curIdx = int(curPoint[0])

        point.append((curPoint[1],curPoint[-1]))

    return data


# Reads results generated by wapiti and CRF++ tools
def readResult(resPath,testIdPath,model):
    vec = readFile(testIdPath)
    # vec = vec[1:]
    testIds = [ele.strip().split('#')[0] for ele in vec]
    sentAspectsPred = {}

    if model == 'MaxEnt':
        resData = readMax(resPath)
    else:
        resData = readData(resPath)

    for idx, resTag in enumerate(resData):
        dataKey = testIds[idx]
        curTerm = ''
        curTermList = []
        for word, wordTag in resTag:
            if wordTag == 'TRUE':
                if curTerm:
                    curTerm = curTerm + " " + word
                else:
                    curTerm = word
            else:
                if curTerm:
                    curTermList.append(curTerm)
                curTerm = ''

        sentAspectsPred[dataKey] = curTermList
    return sentAspectsPred


# Given a word-level classifier, generates list of predicted aspect terms in a sentence
def predict(tagger,testData,testIdPath):
    vec = readFile(testIdPath)
    # vec = vec[1:]
    testIds = [ele.strip().split('#')[0] for ele in vec]
    sentAspectsPred = {}

    for idx, testPoint in enumerate(testData):
        dataKey = testIds[idx]
        dataTag = tagger.tag(testPoint)
        curTerm = ''
        curTermList = []

        # If multiple words are labeled as aspect terms in succession, they are combined into a single aspect term.
        for word, wordTag in dataTag:
            if wordTag == 'TRUE':
                if curTerm:
                    curTerm = curTerm + " " + word
                else:
                    curTerm = word
            else:
                if curTerm:
                    curTermList.append(curTerm)
                curTerm = ''

        sentAspectsPred[dataKey] = curTermList

    return sentAspectsPred


# Baseline model training function
# Computes the number of times each word was labeled as aspect and non-aspect
def BaseTrain(trainData):
    tagger = {}

    for point in trainData:
        for word,wordTag in point:
            if word not in tagger:
                tagger[word] = {'TRUE':0,'FALSE':0}
            tagger[word][wordTag] += 1
    
    return tagger


# Baseline model prediction function
# Computes most common label in training dataset for each word
def BasePred(tagger,testData,testIdPath):
    vec = readFile(testIdPath)
    # vec = vec[1:]
    testIds = [ele.strip().split('#')[0] for ele in vec]
    sentAspectsPred = {}
    for idx, testPoint in enumerate(testData):
        dataKey = testIds[idx]
        dataTag = []
        for testword in testPoint:
            if testword in tagger:
                if tagger[testword]['TRUE'] >= tagger[testword]['FALSE']:
                    dataTag.append((testword,'TRUE'))
                else:
                    dataTag.append((testword,'FALSE'))
            else:
                dataTag.append((testword,'FALSE'))
        
        curTerm = ''
        curTermList = []

        # If multiple words are labeled as aspect terms in succession, they are combined into a single aspect term.
        for word, wordTag in dataTag:
            if wordTag == 'TRUE':
                if curTerm:
                    curTerm = curTerm + " " + word
                else:
                    curTerm = word
            else:
                if curTerm:
                    curTermList.append(curTerm)
                curTerm = ''

        sentAspectsPred[dataKey] = curTermList
    return sentAspectsPred


# Reads gold labels from data 
def readLabel(goldPath):
    vec = readFile(goldPath)
    sentAspectsGold = {}
    for goldData in vec:
        curData = goldData.strip().split("#")
        dataKey = curData[0]
        if curData[-1] == '':
            curTermList = []
        else:
            curTermList = curData[-1].split("&")
        sentAspectsGold[dataKey] = curTermList
    return sentAspectsGold


# Generates precision and recall metrics from predicted and gold labels
def metrics(sentAspectsPred,sentAspectsGold):
    TP = 0
    PD = 0
    RD = 0

    for dataKey in sentAspectsPred:
        if dataKey not in sentAspectsGold:
            print("What?")
        predList = sentAspectsPred[dataKey]
        goldList = sentAspectsGold[dataKey]

        for pred in predList:
            for gold in goldList:
                if pred == gold:
                    TP+=1
        
        PD += len(predList)
        RD += len(goldList)

    precision = TP/PD
    recall = TP/RD
    F1_score = 2*precision*recall/(precision+recall)

    return precision,recall,F1_score

# Prints metrics 
def printMetrics(precision,recall,F1_score):
    print("Precision",precision)
    print("Recall",recall)
    print("F1",F1_score)


# HMM model 
# Loads data, trains and tests model
def HMM(trainPath,testPath,testIdPath,crossVal=False):

    trainData = readData(trainPath,True)
    testData = readData(testPath,False)
    tagger = hmm.HiddenMarkovModelTagger.train(trainData)

    sentAspectsPred = predict(tagger,testData,testIdPath)
    sentAspectsGold = readLabel(goldPath)

    precision, recall, F1_score = metrics(sentAspectsPred,sentAspectsGold)
    if not crossVal:
        printMetrics(precision,recall, F1_score)
    return precision, recall, F1_score
    

# CRF Model 
# Model is trained using CRF++ tool
# Loads model predictions, computes metrics
def CRF(resPath,testIdPath,crossVal = False):
    # resPath = outPath + sys.argv[2]
    sentAspectsPred = readResult(resPath,testIdPath,'CRF')
    sentAspectsGold = readLabel(goldPath)

    precision, recall, F1_score = metrics(sentAspectsPred,sentAspectsGold)
    
    if not crossVal:
        printMetrics(precision,recall,F1_score)  
    return precision,recall, F1_score
# MEMM Model 
# Model is trained using Wapiti tool
# Loads model predictions, computes metrics
def MEMM(resPath,testIdPath,crossVal = False):
    # resPath = outPath + sys.argv[2]
    sentAspectsPred = readResult(resPath,testIdPath,'MEMM')
    sentAspectsGold = readLabel(goldPath)

    precision, recall, F1_score = metrics(sentAspectsPred,sentAspectsGold)
    
    if not crossVal:
        printMetrics(precision,recall,F1_score)  
    return precision,recall, F1_score

# MaxEnt Model 
# Model is trained using Wapiti tool
# Loads model predictions, computes metrics
def MaxEnt(resPath,testIdPath,crossVal = False):
    # resPath = outPath + sys.argv[2]
    sentAspectsPred = readResult(resPath,testIdPath,'MaxEnt')
    sentAspectsGold = readLabel(goldPath)

    precision, recall, F1_score = metrics(sentAspectsPred,sentAspectsGold)
    
    if not crossVal:
        printMetrics(precision,recall, F1_score) 

    return precision,recall,F1_score


# Baseline Model
# Loads data, trains and tests model
def Baseline(trainPath,testPath,testIdPath,crossVal = False):
    trainData = readData(trainPath,True)
    testData = readData(testPath,False)
    tagger = BaseTrain(trainData)
    sentAspectsPred = BasePred(tagger,testData,testIdPath)
    sentAspectsGold = readLabel(goldPath)

    precision, recall, F1_score = metrics(sentAspectsPred,sentAspectsGold)
    
    if not crossVal:
        printMetrics(precision,recall,F1_score)

    return precision,recall,F1_score


if __name__ == '__main__':
    crossVal = True
    if crossVal:
        precisionSum,recallSum,F1_scoreSum = 0, 0, 0
        for ele in range(3):
            
            valtestIdPath = devIdPath + str((ele+2)%3) + '.txt'
            valtrainPath = devTrainPath+str(ele)+str((ele+1)%3)+'.txt'
            valtestPath = devTestPath+str((ele+2)%3)+'.txt'
            
            if sys.argv[1] == 'baseline':
                precision, recall, F1_score = Baseline(valtrainPath,valtestPath,valtestIdPath,True)

            elif sys.argv[1] == 'CRF':
                valresPath = outPath+sys.argv[2] +str((ele+2)%3)+'.txt'
                precision, recall, F1_score =  CRF(valresPath,valtestIdPath,True)

            elif sys.argv[1] == 'HMM':
                precision, recall, F1_score = HMM(valtrainPath,valtestPath,valtestIdPath,True)
            
            elif sys.argv[1] == 'MEMM':
                valresPath = outPath+sys.argv[2] +str((ele+2)%3)+'.txt'
                precision, recall, F1_score = MEMM(valresPath,valtestIdPath,True)

            elif sys.argv[1] == 'MaxEnt':
                valresPath = outPath+sys.argv[2] +str((ele+2)%3)+'.txt'
                precision, recall, F1_score = MaxEnt(valresPath,valtestIdPath,True)

            precisionSum += precision
            recallSum += recall
            F1_scoreSum += F1_score
            print("Precision",precision)
            print("Recall",recall)
            print("F1_score",F1_score)
            print("")
        print("Precision",precisionSum/3)
        print("Recall",recallSum/3)
        print("F1_score",F1_scoreSum/3)
    else:
        resPath = outPath+sys.argv[2] 
        if sys.argv[1] == 'baseline':
            precision, recall, F1_score = Baseline(trainPath,testPath,valtestIdPath,False)

        elif sys.argv[1] == 'CRF':
            precision, recall, F1_score =  CRF(resPath,testIdPath,False)

        elif sys.argv[1] == 'HMM':
            precision, recall, F1_score = HMM(trainPath,testPath,testIdPath,False)
        
        elif sys.argv[1] == 'MEMM':
            precision, recall, F1_score = MEMM(resPath,testIdPath,False)

        elif sys.argv[1] == 'MaxEnt':
            precision, recall, F1_score = MaxEnt(resPath,testIdPath,False)
            




