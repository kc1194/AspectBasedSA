import os 
import sys

os.system('python processing.py')

os.system('python runTools.py CRF')
os.system('python runTools.py MEMM')
os.system('python runTools.py MaxEnt')

os.system('python aspectExtraction.py HMM')
os.system('python aspectExtraction.py baseline')
os.system('python aspectExtraction.py CRF crf++_pred.txt')
os.system('python aspectExtraction.py MEMM memm_pred.txt')
os.system('python aspectExtraction.py MaxEnt maxent_pred.txt')

os.system('python sentimentPredict.py SVM')
os.system('python sentimentPredict.py logreg')
os.system('python sentimentPredict.py nn')
os.system('python sentimentPredict.py adaboost')