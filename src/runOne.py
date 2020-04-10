import os 
import sys

os.system('python processing.py')

if ((sys.argv[1] == 'CRF') or (sys.argv[1] == 'MEMM') or (sys.argv[1] == 'MaxEnt')):
    os.system('python runTools.py '+sys.argv[1])
    os.system('python aspectExtraction.py '+sys.argv[1]+' '+sys.argv[2])
    os.system('python sentimentPredict.py '+sys.argv[3])

else:
    os.system('python aspectExtraction.py '+sys.argv[1])
    os.system('python sentimentPredict.py '+sys.argv[2])
# os.system('python runTools.py ')

