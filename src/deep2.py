import keras
from keras import Sequential
from keras.layers import Embedding,LSTM,Dense,Dropout,SimpleRNN,TimeDistributed,InputLayer,Activation, Attention
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import sys
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
import string

# Data Paths 
trainFile = '../output/aspectTrain.txt'
testFile = '../output/aspectTest.txt'
goldFile = '../data/aspectTerm.txt'
devFile = '../output/aspectDev'

outPath = "../output/"
trainPath = outPath+"parseProcessedTrain"
testPath = outPath+"parseProcessedTest"
deepOutput = outPath+"deepATE"

# Hyperparameters
windowSize = 20

# Function for reading a file
def readFile(filPath):
    fil = open(filPath)
    vec = fil.readlines()
    fil.close()
    return vec

def processSent(sent):
    exclude = set(string.punctuation)
    sent = ''.join(ch for ch in sent if ch not in exclude)
    
    if sent[-1] == u'।':
        sent = sent[:-1]
    
    return sent

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
    vocab = {}
    # invmapper = {}
    vec = [ele.strip().split('#')[1] for ele in vec]
    vec = [processSent(ele).split(' ') for ele in vec]
    counter = 1
    for sent in vec:
        for word in sent:
            # if word not in mapper:
            #     mapper[word] = counter
            #     invmapper[counter] = word
            #     counter+=1
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    num_words = len([ele for ele in vocab if vocab[ele]>3])

    for sent in vec:
        for word in sent:
            if word not in mapper:
                if vocab[word] > 3:
                    mapper[word] = counter
                # invmapper[counter] = word
                    counter+=1
                else:
                    mapper[word] = num_words+1
            
    
    return mapper,num_words

def ATEMapping():
    vec = readFile(goldFile)
    mapper = {}
    invmapper = {}
    vec = [ele.strip().split('#')[1] for ele in vec]
    vec = [processSent(ele).split(' ') for ele in vec]
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
            for curElement in processSent(window).split(" "):
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
                Labels.append(np.array([1,0,0,0]))
            elif label == 'neg':
                Labels.append(np.array([0,1,0,0]))
            elif label == 'neu':
                Labels.append(np.array([0,0,1,0]))
            else:
                Labels.append(np.array([0,0,0,1]))
    
    return Data,Labels
    # if train:
    #     return Data,np.array(Labels)
    # else:
    #     return Data,np.array(Labels)

def readData(dataPath,mapper,train=True):
    vec = readFile(dataPath)
    data = []
    point = [] 
    labelpoint = []
    labels = []

    for ele in vec:
        if (ele=='\n'):
            data.append(point)
            if train:
                labels.append(np.array(labelpoint))
            point = []
            labelpoint = []
            continue
        curPoint = ele.strip().split("\t")
        if curPoint[1] in mapper:
            point.append(mapper[curPoint[1]])
        else:
            point.append(0)
        if train:
            if curPoint[-1] == 'TRUE':
                labelpoint.append([1,0])
            else:
                labelpoint.append([0,1])

    if train:
        return data,np.array(labels)
    else:
        return data

def sentModel(num_words):
    model = Sequential()
    model.add(Embedding(num_words+2, 10, input_length=20)) 
    model.add(LSTM(32)) 
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax')) 
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['categorical_accuracy']) 
    print(model.summary()) 
    return model

def fitSentModel(trainData,trainLabels,testData,goldLabels,model):
    trainPad = pad_sequences(trainData,maxlen=20,padding='post')
    testPad = pad_sequences(testData,maxlen=20,padding='post')

    model.fit(trainPad,trainLabels,batch_size=32,epochs=10,validation_data=(testPad,goldLabels),verbose=2)
    predictions = model.predict(testPad)
    return predictions

def sentMetrics(predictions,goldLabels):
    predVal = []
    goldVal = []

    vals = ['pos','neg','neu','con']
    for idx, pred in enumerate(predictions):
        predId = np.argmax(pred)
        goldId = np.argmax(goldLabels[idx])
        predVal.append(vals[predId])
        goldVal.append(vals[goldId])
    return accuracy_score(predVal,goldVal),confusion_matrix(goldVal,predVal,labels=["pos", "neg", "neu","con"])

def runSent(crossVal = True):
    mapper,num_words = createMapping()

    if not crossVal:
        model = sentModel(num_words)
        
        trainData,trainLabels = extractData(trainFile,mapper,True)
        testData,goldLabels = extractData(testFile,mapper,False)
        trainLabels = np.array(trainLabels)

        predictions = fitSentModel(trainData, trainLabels, testData, goldLabels, model)
        acc, conf_matrix = sentMetrics(predictions, goldLabels)
        print("Accuracy",acc)
        print("Confusion Matrix",conf_matrix)

        
    
    else:

        mean_acc = 0

        for curItr in range(3):
            model = sentModel(num_words)
            trainData1,trainLabels1 = extractData(devFile+str(curItr)+'.txt',mapper,True)
            trainData2,trainLabels2 = extractData(devFile+str((curItr+1)%3)+'.txt',mapper,True)
            testData,goldLabels = extractData(devFile+str((curItr+2)%3)+'.txt',mapper,False)

            trainData = trainData1+trainData2
            trainLabels = np.array(trainLabels1+trainLabels2)
            goldLabels = np.array(goldLabels)

            predictions = fitSentModel(trainData, trainLabels, testData, goldLabels, model)
            acc, conf_matrix = sentMetrics(predictions, goldLabels)
            
            print("Accuracy",acc)
            print("Confusion Matrix",conf_matrix)
            print("")
            mean_acc+= acc
            if curItr==0:
                totalConf = conf_matrix
            else:
                totalConf+=conf_matrix

        print("Mean Accuracy",mean_acc/3)
        print("Total Conf",totalConf)



def ATEModel(MAX_LENGTH,num_words):
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

    return model

def fitATEModel(trainData,trainLabels,testData,model,MAX_LENGTH):
    trainPad = pad_sequences(trainData,maxlen=MAX_LENGTH,padding='post')
    testPad = pad_sequences(testData,maxlen=MAX_LENGTH,padding='post')
    trainLabelsPad = pad_sequences(trainLabels,maxlen=MAX_LENGTH,padding='post')
    # testLabelsPad = pad_sequences(testLabels,maxlen=MAX_LENGTH,padding='post')

    model.fit(trainPad,trainLabelsPad,batch_size=32,epochs=20,verbose=2)
    predictions = model.predict(testPad)

    return predictions

def ATEOutput(fil,predictions,outputData):
    predVal = ['TRUE','FALSE']
    # print(predictions)
    outIdx = 0
    for predict in predictions:
        for predIdx,curPred in enumerate(predict):
            if outputData[outIdx] == '\n':
                if predIdx >0:
                    fil.write('\n')
                    outIdx+=1 
                    break
                else:
                    fil.write('\n')
                    outIdx+=1
                    continue
            try:
                curOut = outputData[outIdx].strip()+'\t'+predVal[1-int(curPred[0]>0.6)]+'\n'
            except:
                print(outIdx)
                print(len(outputData))
                print(len(predVal))
                print(curPred)
                print(np.argmax(curPred))
                sys.exit()
            fil.write(curOut)
            outIdx+=1

def runATE(crossVal = True):
    mapper,invmapper = ATEMapping()
    num_words = len([ele for ele in mapper])

    

    #model
    if not crossVal:
        trainData,trainLabels = readData(trainPath+'.txt',mapper)
        testData = readData(testPath+'.txt',mapper,False)
        MAX_LENGTH = len(max(trainData, key=len))

        model = ATEModel(MAX_LENGTH,num_words)
        predictions = fitATEModel(trainData,trainLabels,testData,model,MAX_LENGTH)
        outputData = readFile(testPath+'.txt')
        fil = open(deepOutput+'.txt','w')
        ATEOutput(fil,predictions,outputData)
    
    else:
        for curItr in range(3):
            trainData,trainLabels = readData(trainPath+str(curItr)+str((curItr+1)%3)+'.txt',mapper)
            testData = readData(testPath+str((curItr+2)%3)+'.txt',mapper,False)
            MAX_LENGTH = len(max(trainData, key=len))

            model = ATEModel(MAX_LENGTH,num_words)
            predictions = fitATEModel(trainData,trainLabels,testData,model,MAX_LENGTH)
            outputData = readFile(testPath+str((curItr+2)%3)+'.txt')
            fil = open(deepOutput+str((curItr+2)%3)+'.txt','w')
            ATEOutput(fil,predictions,outputData)

    
    # TP,FP,FN,TN = 0,0,0,0
    # for idx,predict in enumerate(predictions):
    #     gold = testLabelsPad[idx]
    #     for jdx,goldTag in enumerate(gold):
    #         if (goldTag[0] == 0) and (goldTag[1] == 0):
    #             break
    #         if goldTag[0] == 1:
    #             if predict[jdx][0]>=predict[jdx][1]:
    #                 TP+=1
    #             else:
    #                 FN+=1
    #         else:
    #             if predict[jdx][0]>=predict[jdx][1]:
    #                 FP+=1
    #             else:
    #                 TN+=1
    # print(TP,FP,TN,FN)
    # print("Recall",TP/(TP+FN))
    # print("Precision",TP/(TP+FP))
    # print("Accuracy",(TP+TN)/(TP+TN+FP+FN))
    # print(confusion_matrix(predictions.argmax(axis=1),testPad.argmax(axis=1)))




runSent()