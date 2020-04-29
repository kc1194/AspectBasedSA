import os
import sys

# Runs CRF++ tool in train and test mode
def runCRF(crossVal):
    if not crossVal:
        os.system('./../tools/CRF++-0.58/crf_learn ../src/patternFile.txt ../output/parseProcessedTrain.txt ../output/crf++_model_vocab')
        os.system('./../tools/CRF++-0.58/crf_test -m ../output/crf++_model_vocab ../output/parseProcessedTest.txt > ../output/crf++_pred_vocab.txt')
    else:
        for ele in range(3):
            os.system('./../tools/CRF++-0.58/crf_learn ../src/patternFile.txt ../output/parseProcessedTrain'+str(ele)+str((ele+1)%3)+'Eng.txt ../output/crf++_modelEng'+str(ele)+str((ele+1)%3))
            os.system('./../tools/CRF++-0.58/crf_test -m ../output/crf++_modelEng'+str(ele)+str((ele+1)%3)+' ../output/parseProcessedTest'+str((ele+2)%3)+'Eng.txt > ../output/crf++_predEng'+str((ele+2)%3)+'.txt')

# Runs Wapiti tool in train and test mode for either MEMM or MaxEnt models 
def runWapiti(mode,crossVal):
    if mode == 'MEMM':
        if not crossVal:
            os.system('./../tools/wapiti-1.5.0/wapiti train -p ../src/patternFile.txt -T memm -w 20 ../output/parseProcessedTrain.txt ../output/memm_model')
            os.system('./../tools/wapiti-1.5.0/wapiti label -m ../output/memm_model ../output/parseProcessedTest.txt ../output/memm_pred.txt')
        else:
            for ele in range(3):
                os.system('./../tools/wapiti-1.5.0/wapiti train -p ../src/patternFile.txt -T memm -w 20 ../output/parseProcessedTrain'+str(ele)+str((ele+1)%3)+'.txt ../output/memm_model'+str(ele)+str((ele+1)%3))
                os.system('./../tools/wapiti-1.5.0/wapiti label -m ../output/memm_model'+str(ele)+str((ele+1)%3)+' ../output/parseProcessedTest'+str((ele+2)%3)+'.txt ../output/memm_pred'+str((ele+2)%3)+'.txt')
    else:
        if not crossVal:
            os.system('./../tools/wapiti-1.5.0/wapiti train -p ../src/patternFile.txt -T maxent --me -o 20 -w 20 ../output/parseProcessedTrain.txt ../output/maxent_model')
            os.system('./../tools/wapiti-1.5.0/wapiti label -m ../output/maxent_model --me ../output/parseProcessedTest.txt ../output/maxent_pred.txt')
        else:
            for ele in range(3):
                os.system('./../tools/wapiti-1.5.0/wapiti train -p ../src/patternFile.txt -T maxent --me -o 20 -w 20 ../output/parseProcessedTrain'+str(ele)+str((ele+1)%3)+'.txt ../output/maxent_model'+str(ele)+str((ele+1)%3))
                os.system('./../tools/wapiti-1.5.0/wapiti label -m ../output/maxent_model'+str(ele)+str((ele+1)%3)+' --me ../output/parseProcessedTest'+str((ele+2)%3)+'.txt ../output/maxent_pred'+str((ele+2)%3)+'.txt')


if __name__ == '__main__':
    
    if sys.argv[1] == 'CRF':
        runCRF(bool(sys.argv[2]))
    
    elif sys.argv[1] == 'MEMM':
        runWapiti(sys.argv[1],bool(sys.argv[2]))
    
    elif sys.argv[1] == 'MaxEnt':
        runWapiti(sys.argv[1],bool(sys.argv[2]))
