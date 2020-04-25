# Imports
import sys
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# Data Paths 
trainFile = '../output/aspectTrain.txt'
testFile = '../output/aspectTest.txt'
goldFile = '../data/aspectTerm.txt'
devFile = '../output/aspectDev'

# Hyperparameters
windowSize = 20

# Function for reading a file
def readFile(filPath):
    fil = open(filPath)
    vec = fil.readlines()
    fil.close()
    return vec

# Function for extracting window of text around an aspect term
def extractWindow(aspectTerm,wordList):
    aspectTermWords = aspectTerm.split(' ')
    aspectSize = len(aspectTermWords)
    remWindowSize = windowSize-aspectSize

    startIndex = 0
    for idx,word in enumerate(wordList):
        if aspectTermWords[0] in word:
            startIndex = idx
            break

    
    windowStart = max(0,startIndex - int(remWindowSize/2))
    windowEnd = min(startIndex+aspectSize+int(remWindowSize/2),len(wordList))
    window = wordList[windowStart:windowEnd]

    return ' '.join(window)

# Function to read gold labels
def extractGoldLabels():
    vec1 = readFile(goldFile)
    vec1 = [ele.strip().split('#') for ele in vec1]
    labelDict = {}
    for ele in vec1:
        if ele[-1] == '':
            continue
        labelList = ele[-2].split('&')
        labelDict[ele[0]] = labelList
    return labelDict

def baseTrain(filPath):
    vec1 = readFile(filPath)
    vec1 = [ele.strip().split('#') for ele in vec1]
    labelCountDict = {}
    for ele in vec1:
        if ele[-1] == '':
            continue
        aspectTermList = ele[-1].split('&')
        labelList = ele[-2].split('&')

        for idx,aspectTerm in enumerate(aspectTermList):
            if aspectTerm in labelCountDict:
                if labelList[idx] in labelCountDict[aspectTerm]:
                    labelCountDict[aspectTerm][labelList[idx]] += 1
                else:
                    labelCountDict[aspectTerm][labelList[idx]] = 1
            else:
                labelCountDict[aspectTerm] = {}
                labelCountDict[aspectTerm][labelList[idx]] = 1
    return labelCountDict

def basePred(filPath,labelCountDict):
    vec1 = readFile(filPath)
    vec1 = [ele.strip().split('#') for ele in vec1]
    labelDict = extractGoldLabels()
    goldLabels = []
    predLabels = []
    for ele in vec1:
        if ele[-1] == '':
            continue
        aspectTermList = ele[-1].split('&')
        for aspectIdx,aspectTerm in enumerate(aspectTermList):
            if aspectTerm in labelCountDict:
                curMax = 0
                curKey = 'neu'
                for key in labelCountDict[aspectTerm]:
                    if labelCountDict[aspectTerm][key] > curMax:
                        curKey = key
                        curMax = labelCountDict[aspectTerm][key]
                predLabels.append(curKey)
            else:
                predLabels.append('neu')

                

            
            labelGold = labelDict[ele[0]][aspectIdx]
            goldLabels.append(labelGold)
    
    return predLabels,goldLabels

def addToDict(myDict,outerKey,innerKey):
    if outerKey in myDict:
        if innerKey in myDict[outerKey]:
            myDict[outerKey][innerKey] += 1
        else:
            myDict[outerKey][innerKey] = 1
    else:
        myDict[outerKey] = {}
        myDict[outerKey][innerKey] = 1

def mergeDictHelper(dictMerged,dictA,dictB):
    for key in dictA:
        if key in dictB:
            dictMerged[key] = dictA[key]+dictB[key]
        else:
            dictMerged[key] = dictA[key]
    for key in dictB:
        if key not in dictA:
            dictMerged[key] = dictB[key]

def mergeDict(dictMerged,dictA,dictB):
    for key in dictA:
        if key in dictB:
            dictMerged[key] = {}
            mergeDictHelper(dictMerged[key],dictA[key],dictB[key])
        else:
            dictMerged[key] = dictA[key]
    for key in dictB:
        if key not in dictA:
            dictMerged[key] = dictB[key]

# Function to read training and testing data
def extractData(filPath,train=True):
    # if train:
    #     vec1 = readFile(trainFile)
    # else:
    #     vec1 = readFile(testFile)

    # labelCountDict = {}
    vec1 = readFile(filPath)
    # vec1 = vec1[1:]
    vec1 = [ele.strip().split('#') for ele in vec1]

    Data = []
    Labels = []
    if train:
        SOCounter = {}
        MCLCounter = {}
    if not train:
        labelDict = extractGoldLabels()
    for ele in vec1:
        if ele[-1] == '':
            continue
        aspectTermList = ele[-1].split('&')
        if train:
            labelList = ele[-2].split('&')
            # baseCounts(aspectTermList,labelList, labelCountDict)
        wordList = ele[1].split(' ')
        for aspectIdx,aspectTerm in enumerate(aspectTermList):
            window = extractWindow(aspectTerm,wordList)
            Data.append(window)
            if train:
                label = labelList[aspectIdx]
                addToDict(MCLCounter,aspectTerm,label)
                for contWord in window:
                    addToDict(SOCounter,contWord,label)
            else:
                label = labelDict[ele[0]][aspectIdx]
            Labels.append(label)
    
    if train:
        return Data,Labels,MCLCounter,SOCounter
    else:
        return Data,Labels
    
# Function for running the classifier
# Reads data, converts to features, trains and tests classifier
def runClassifier(clfName,crossVal=False):
    
    if clfName == 'SVM':
            clf = svm.SVC(kernel='linear')
    elif clfName == 'logreg':
        clf = LogisticRegression()
    elif clfName == 'nn':
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    elif clfName == 'NB':
        clf = MultinomialNB()
    else:
        clf = AdaBoostClassifier(n_estimators=300, random_state=0)

    if crossVal:
        itr = range(3)
    else:
        itr = range(1)
    
    acc = 0
    devFiles = [devFile+str(ele)+'.txt' for ele in range(3)]
    for ele in itr:
        if not crossVal:
            trainData,trainLabels = extractData(trainFile,True)
            testData,goldLabels = extractData(testFile,False)
        else:
            trainData1,trainLabels1,trainMCLCounter1,trainSOCounter1 = extractData(devFiles[ele],True)
            trainData2,trainLabels2,trainMCLCounter2,trainSOCounter2 = extractData(devFiles[(ele+1)%3],True)
            trainData,trainLabels = trainData1+trainData2,trainLabels1+trainLabels2
            testData,goldLabels = extractData(devFiles[(ele+2)%3],False)
        
        trainMCLCounter = {}
        trainSOCounter = {}
        mergeDict(trainSOCounter,trainSOCounter1,trainSOCounter2)
        mergeDict(trainMCLCounter,trainMCLCounter1,trainMCLCounter2)

        print(len(trainData))
        sys.exit(0)
        featureTransform = TfidfVectorizer()
        featureTransform = featureTransform.fit(trainData)

        trainFeatures = featureTransform.transform(trainData)
        testFeatures = featureTransform.transform(testData)

        # Adding features


        clf.fit(trainFeatures,trainLabels)
        predLabels = clf.predict(testFeatures)

        acc += accuracy_score(predLabels,goldLabels)
        print("Accuracy",accuracy_score(predLabels,goldLabels))
        print("Confusion",confusion_matrix(goldLabels, predLabels, labels=["pos", "neg", "neu","con"]))
        if ele == 0:
            conf = confusion_matrix(goldLabels, predLabels, labels=["pos", "neg", "neu","con"])
        else:
            conf += confusion_matrix(goldLabels, predLabels, labels=["pos", "neg", "neu","con"])

    if crossVal:
        acc = acc/3

    print("")
    print("Accuracy",acc)
    print("Confusion",conf)


runClassifier(sys.argv[1],bool(sys.argv[2]))
# labelCountDict = baseTrain(trainFile)
# predLabels,goldLabels = basePred(testFile,labelCountDict)
# print("Accuracy",accuracy_score(predLabels,goldLabels))
# print(predLabels)
# print(labelCountDict)
# runClassifier(sys.argv[1],False)


    



        

