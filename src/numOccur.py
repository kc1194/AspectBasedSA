import string
import numpy as np


outPath = '../output/'
devFileTrain = outPath+'parseProcessedTrain'
devFileTest = outPath+'parseProcessedTest'

def processSent(sent):
    exclude = set(string.punctuation)
    sent = ''.join(ch for ch in sent if ch not in exclude)
    
    if sent[-1] == u'।':
        sent = sent[:-1]
    
    return sent


# print(len(aspectSet))
# print(len(aspectSet-singleSet))
# print(len(singleSet))
# print(len(singleSet-aspectSet))
# print([ele for ele in singleSet])
# print(nonAspectCount['लूमिया'])



def readFile(filPath):
    fil = open(filPath)
    vec = fil.readlines()
    fil.close()
    return vec

def aspectSetComp(curItr):
    fil = open('../data/aspectTerm.txt')
    vec = fil.readlines()
    fil.close()
    np.random.seed(1)
    arr = np.arange(len(vec))
    np.random.shuffle(arr)

    devIdx1 = int(len(vec)/3)
    devIdx2 = int(2*len(vec)/3)

    if curItr == 0:
        vec = vec[:devIdx2]
    elif curItr ==1:
        vec = vec[devIdx1:]
    else:
        vec = vec[devIdx2:]+vec[:devIdx1]
    # indices = [0,devIdx1,devIdx2,len(vecData)]

    
    
    
    aspectCount = {}
    nonAspectCount = {}

    for ele in vec:
        comp = ele.strip().split('#')
        curSent = comp[1]
        curSentProc = processSent(curSent)
        curWords = curSentProc.split(" ")
        if comp[-1] == '':
            continue
        curAspectTerms = comp[-1].split('&')
        curAspectWords = set([])
        for aspect in curAspectTerms:
            aspectProc = processSent(aspect)
            for aspectWord in aspectProc.split(' '):
                if aspectWord in aspectCount:
                    aspectCount[aspectWord] += 1
                else:
                    aspectCount[aspectWord] = 1
                curAspectWords.add(aspectWord)
        for word in curWords:
            if True:
                if word in nonAspectCount:
                    nonAspectCount[word] += 1
                else:
                    nonAspectCount[word] = 1

    aspectSet = set([ele for ele in aspectCount])
    # singleSet = set([ele for ele in nonAspectCount if nonAspectCount[ele]<=1])
    return aspectSet


def numFeatures(featurePath,numFeaturePath,curItr,train=True):
    vec1 = readFile(featurePath)
    vec3 = [ele.strip().split("\t") for ele in vec1]
    vec4 = [ele+[ele[-1]] for ele in vec3]
    # import ipdb; ipdb.set_trace()
    aspectSet = aspectSetComp(curItr)
    for idx,ele in enumerate(vec4):
        if ele[0] == '':
            continue
        if (ele[1] in aspectSet):
            # print(idx)
            if train:
                ele[-2] = 'TRUE'
            else:
                ele[-1] = 'TRUE'
        else:
            if train:
                ele[-2] = 'FALSE'
            else:
                ele[-1] = 'FALSE'

    fil3 = open(numFeaturePath,"w")
    for ele in vec4:
        fil3.write("\t".join(ele)+'\n')
    fil3.close()

for ele in range(3):
    
    if ele<2:
        numFeatures(devFileTrain+str(ele)+str((ele+1)%3)+".txt",devFileTrain+str(ele)+str((ele+1)%3)+"Num.txt",ele,True)

    else:
        numFeatures(devFileTrain+str(ele)+str((ele+1)%3)+".txt",devFileTrain+str(ele)+str((ele+1)%3)+"Num.txt",ele,True)

    numFeatures(devFileTest+str(ele)+'.txt',devFileTest+str(ele)+'Num.txt',ele,False)