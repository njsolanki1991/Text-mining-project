# Libraries
import zipfile
import json
import pandas as pd
import os

# Extract zip file in the relative provided
def ExtractZipFile(Path):    
    # Make sure that the zip file name 'pan20-authorship-verification-training-small.zip' is placed in the relative path
    with zipfile.ZipFile(Path+ '/pan20-authorship-verification-training-small.zip', 'r') as zip_ref:
        zip_ref.extractall(Path) # Exptract in the relative path as provided

# Opening Files and loading data
def LoadAllData(path):
    with open(path, 'r') as json_file:
        JsonList = list(json_file)
    return JsonList

# Random shuffling of data
def RandomShuffling(TruthList,DataList):
    dataFrame=pd.DataFrame(columns=['Truth', 'Data'])
    dataFrame=pd.DataFrame(TruthList)
    dataFrame.rename(columns = {0:'Truth'}, inplace = True)
    dataFrame['Data']=pd.DataFrame(DataList)
    # Shuffling of data
    dataFrame=dataFrame.sample(frac=1)
    return dataFrame

# Creating jsonl files from listin the folder
def CreateJSONLFiles(folderPath,fileName, data):
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)
    with open(folderPath+'/'+fileName, 'w') as outfile:
        for entry in data:            
            json.dump(json.loads(entry[:-1]), outfile)
            outfile.write('\n')

# Path should be relative path and folder called Datasets should exist in same path of this code
def main(Path):
    # Extract zip File from relative path
    ExtractZipFile(Path)
    # Get data from the provided path and shuffle them
    GroundTruthJsonlist=LoadAllData(Path+'/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl')
    DataJsonlist=LoadAllData(Path + '/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl')
    dataFrame=RandomShuffling(GroundTruthJsonlist,DataJsonlist)
    GroundTruthJsonlistNew=dataFrame['Truth'].values.tolist()
    DataJsonlistNew=dataFrame['Data'].values.tolist()

    # Dividing into training and test in 90:10 respectively
    trainingCut=int(0.9 * len(DataJsonlistNew))
    trainingData, testData = DataJsonlistNew[:trainingCut], DataJsonlistNew[trainingCut:]
    groundTruthTrainingData, groundTruthTestData = GroundTruthJsonlistNew[:trainingCut], GroundTruthJsonlistNew[trainingCut:]    

    trainingDissimilarityMethodPath= Path + '/pan20-authorship-verification/DissimilarityMethod/training'    
    testDissimilarityMethodPath= Path + '/pan20-authorship-verification/DissimilarityMethod/test'
    
    # Creating 90: 10 split
    CreateJSONLFiles(trainingDissimilarityMethodPath,'pairs.jsonl',trainingData)
    CreateJSONLFiles(trainingDissimilarityMethodPath,'truth.jsonl',groundTruthTrainingData)
    CreateJSONLFiles(testDissimilarityMethodPath,'pairs.jsonl',testData)
    CreateJSONLFiles(testDissimilarityMethodPath,'truth.jsonl',groundTruthTestData)
   
        
    # Printing of datalength
    print('Training Data length:',len(trainingData), len(groundTruthTrainingData))    
    print('Test Data length:',len(testData), len(groundTruthTestData))
    print('Total Data length:',len(DataJsonlist))
            
            
            
            
            
            
            