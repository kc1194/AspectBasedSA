import os
import sys

# Runs CRF++ tool in train and test mode
def runCRF():
    os.system('./../tools/CRF++-0.58/crf_learn ../src/patternFile.txt ../output/parseProcessedTrain.txt ../output/crf++_model')
    os.system('./../tools/CRF++-0.58/crf_test -m ../output/crf++_model ../output/parseProcessedTest.txt > ../output/crf++_pred.txt')

# Runs Wapiti tool in train and test mode for either MEMM or MaxEnt models 
def runWapiti(mode):
    if mode == 'MEMM':
        os.system('./../tools/wapiti-1.5.0/wapiti train -p ../src/patternFile.txt -T memm -w 20 ../output/parseProcessedTrain.txt ../output/memm_model')
        os.system('./../tools/wapiti-1.5.0/wapiti label -m ../output/memm_model ../output/parseProcessedTest.txt ../output/memm_pred.txt')
    else:
        os.system('./../tools/wapiti-1.5.0/wapiti train -p ../src/patternFile.txt -T maxent --me -o 20 -w 20 ../output/parseProcessedTrain.txt ../output/maxent_model')
        os.system('./../tools/wapiti-1.5.0/wapiti label -m ../output/maxent_model --me ../output/parseProcessedTest.txt ../output/maxent_pred.txt')

if __name__ == '__main__':
    
    if sys.argv[1] == 'CRF':
        runCRF()
    
    elif sys.argv[1] == 'MEMM':
        runWapiti(sys.argv[1])
    
    elif sys.argv[1] == 'MaxEnt':
        runWapiti(sys.argv[1])
