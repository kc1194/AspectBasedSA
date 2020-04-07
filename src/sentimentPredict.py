# Imports
import sys
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC

# Data Paths 
trainFile = '../output/aspectTrain.txt'
testFile = '../output/aspectTest.txt'
goldFile = '../data/aspectTerm.txt'

# Hyperparameters
windowSize = 10

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

# Function to read training and testing data
def extractData(train=True):
    if train:
        vec1 = readFile(trainFile)
    else:
        vec1 = readFile(testFile)
    vec1 = vec1[1:]
    vec1 = [ele.strip().split('#') for ele in vec1]

    Data = []
    Labels = []
    if not train:
        labelDict = extractGoldLabels()
    for ele in vec1:
        if ele[-1] == '':
            continue
        aspectTermList = ele[-1].split('&')
        if train:
            labelList = ele[-2].split('&')
        wordList = ele[1].split(' ')
        for aspectIdx,aspectTerm in enumerate(aspectTermList):
            window = extractWindow(aspectTerm,wordList)
            Data.append(window)
            if train:
                label = labelList[aspectIdx]                
            else:
                label = labelDict[ele[0]][aspectIdx]
            Labels.append(label)

    return Data,Labels
    
# Function for running the classifier
# Reads data, converts to features, trains and tests classifier
def runClassifier(clfName):
    trainData,trainLabels = extractData(True)
    testData,goldLabels = extractData(False)
    
    featureTransform = TfidfVectorizer(min_df=0.002)
    featureTransform = featureTransform.fit(trainData)
    
    if clfName == 'SVM':
        clf = svm.SVC(kernel='linear')
    elif clfName == 'logreg':
        clf = LogisticRegression()
    elif clfName == 'nn':
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    else:
        clf = AdaBoostClassifier(n_estimators=300, random_state=0)

    trainFeatures = featureTransform.transform(trainData)
    testFeatures = featureTransform.transform(testData)
 
    clf.fit(trainFeatures,trainLabels)
    predLabels = clf.predict(testFeatures)

    print("Accuracy",accuracy_score(predLabels,goldLabels))


runClassifier(sys.argv[1])


    



        

