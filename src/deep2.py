import keras
from keras import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout,SimpleRNN,TimeDistributed,InputLayer,Activation
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
# Data Paths 
trainFile = '../output/aspectTrain.txt'
testFile = '../output/aspectTest.txt'
goldFile = '../data/aspectTerm.txt'
devFile = '../output/aspectDev'

outPath = "../output/"
trainPath = outPath+"parseProcessedTrain.txt"
testPath = outPath+"parseProcessedTestLabel.txt"

# Hyperparameters
windowSize = 20

# Function for reading a file
def readFile(filPath):
    fil = open(filPath)
    vec = fil.readlines()
    fil.close()
    return vec

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

def createMapping():
    vec = readFile(goldFile)
    mapper = {}
    invmapper = {}
    vec = [ele.strip().split('#')[1] for ele in vec]
    vec = [ele.split(' ') for ele in vec]
    counter = 0
    for sent in vec:
        for word in sent:
            if word not in mapper:
                mapper[word] = counter
                invmapper[counter] = word
                counter+=1
    
    return mapper,invmapper

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

def extractData(filPath,mapper,train=True):
    vec1 = readFile(filPath)
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
            # print(window)
            indices = []
            for curElement in window.split(" "):
                if curElement in mapper:
                    indices.append(mapper[curElement])
                else:
                    indices.append(0)
                    print(curElement)
                    print(window)
                    sys.exit()
            # indices = [mapper[ele] for ele in window]
            Data.append(indices)
            if train:
                label = labelList[aspectIdx]
            else:
                label = labelDict[ele[0]][aspectIdx]
            
            if label == 'pos':
                Labels.append(np.array([1,0,0]))
            elif label == 'neg':
                Labels.append(np.array([0,1,0]))
            else:
                Labels.append(np.array([0,0,1]))
    
    if train:
        return Data,np.array(Labels)
    else:
        return Data,np.array(Labels)

def readData(dataPath,mapper):
    vec = readFile(dataPath)
    data = []
    point = [] 
    labelpoint = []
    labels = []

    for ele in vec:
        if (ele=='\n'):
            data.append(point)
            labels.append(np.array(labelpoint))
            point = []
            labelpoint = []
            continue
        curPoint = ele.strip().split("\t")
        if curPoint[1] in mapper:
            point.append(mapper[curPoint[1]])
        else:
            point.append(0)
        if curPoint[-1] == 'TRUE':
            labelpoint.append([1,0])
        else:
            labelpoint.append([0,1])

    return data,np.array(labels)

def runSent():
    mapper,invmapper = createMapping()
    num_words = len([ele for ele in mapper])

    #model
    model = Sequential()
    model.add(Embedding(num_words, 100, input_length=20)) 
    model.add(LSTM(50)) 
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax')) 
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['categorical_accuracy']) 
    print(model.summary()) 

    trainData,trainLabels = extractData(trainFile,mapper,True)
    testData,goldLabels = extractData(testFile,mapper,False)

    trainPad = pad_sequences(trainData,maxlen=20,padding='post')
    testPad = pad_sequences(testData,maxlen=20,padding='post')

    model.fit(trainPad,trainLabels,batch_size=32,epochs=10,validation_data=(testPad,goldLabels),verbose=2)

def runATE():
    mapper,invmapper = createMapping()
    num_words = len([ele for ele in mapper])

    trainData,trainLabels = readData(trainPath,mapper)
    testData,testLabels = readData(testPath,mapper)

    # print(trainData)
    # print(testData)

    MAX_LENGTH = len(max(trainData, key=len))

    #model
    model = Sequential()
    # model.add(InputLayer(input_shape=(MAX_LENGTH, )))
    model.add(Embedding(num_words, 10)) 
    model.add(LSTM(16,input_shape=(MAX_LENGTH, 10),return_sequences=True)) 
    # model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(2)))
    model.add(Activation('softmax'))
    # model.add(Dense(3, activation='softmax')) 
    model.compile(loss='categorical_crossentropy',optimizer='adam') 
    print(model.summary()) 

    

    trainPad = pad_sequences(trainData,maxlen=MAX_LENGTH,padding='post')
    testPad = pad_sequences(testData,maxlen=MAX_LENGTH,padding='post')
    trainLabelsPad = pad_sequences(trainLabels,maxlen=MAX_LENGTH,padding='post')
    testLabelsPad = pad_sequences(testLabels,maxlen=MAX_LENGTH,padding='post')

    model.fit(trainPad,trainLabelsPad,batch_size=32,epochs=20,validation_data=(testPad,testLabelsPad),verbose=2)
    predictions = model.predict(testPad)
    TP,FP,FN,TN = 0,0,0,0
    for idx,predict in enumerate(predictions):
        gold = testLabelsPad[idx]
        for jdx,goldTag in enumerate(gold):
            if (goldTag[0] == 0) and (goldTag[1] == 0):
                break
            if goldTag[0] == 1:
                if predict[jdx][0]>=predict[jdx][1]:
                    TP+=1
                else:
                    FN+=1
            else:
                if predict[jdx][0]>=predict[jdx][1]:
                    FP+=1
                else:
                    TN+=1
    print(TP,FP,TN,FN)
    print("Recall",TP/(TP+FN))
    print("Precision",TP/(TP+FP))
    print("Accuracy",(TP+TN)/(TP+TN+FP+FN))
    # print(confusion_matrix(predictions.argmax(axis=1),testPad.argmax(axis=1)))




runSent()