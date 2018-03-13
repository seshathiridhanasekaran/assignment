'''
Using a programming language of your choice, implement the following:
1. Accept the number of labels and their distribution from the user, generate a synthetic dataset of
10,000 labels.
example: number of labels: 3
distribution: 30, 40, 30
2. Run stratified 10 fold cross validation on this data generates in 1. implementing:
  * chance baseline
  * majority baseline
3. Compute average micro Precision, Recall, F-measure
'''

from __future__ import division
from random import randrange
import pandas as pd
import numpy as np

numLabels = input("Enter the number of labels:\n")

def getDist(numLabels):
    dist = []
    c= True
    while(c):
        for i in range(0,numLabels):
            val = input("Distribution of "+str(i+1)+" label:")
            dist.append(int(val))

        if sum(dist) != 100:
            print "Wrong distributions entered"
            del dist[:]
            continue
        else:
            break 
    return [d/100 for d in dist]

def getData(dist):
    labels = 10000
    data = []
    for i in range(0,len(dist)):
        val = int(labels*dist[i])
        data.append([i]*val)
    data = [i for list in data for i in list]
    return data

def cross_validation_split(dataset, folds=10):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def chance_baseline(data):
    unique = list(set(data))
    predicted = list()
    for row in data:
        index = randrange(len(unique))
        predicted.append(unique[index])
    return predicted

def majority_baseline(data):
    prediction = max(set(data), key= data.count)
    predicted = [prediction for i in range(len(data))]
    return predicted

def getConfusionMatVal(data, predictions):
    TP=[]
    TN=[]
    FP=[]
    FN=[]

    data = pd.Series(data, name='Actual')
    predicted = pd.Series(predictions, name='Predicted')

    df_confusion = pd.crosstab(data, predicted)
    print df_confusion
    r,c = df_confusion.shape
    sum = df_confusion.values.sum()
    print r,c
    df_confusion = df_confusion.values.tolist()

    for i in range(r):
        for j in range(c):
            if i == j:
                tp = df_confusion[i][j]
                TP.append(tp)
                k = 0
                fn=0
                while(k<c):
                    if k!=i:
                        fn += df_confusion[i][k]
                    k+=1
                FN.append(fn)
                k=0
                fp=0
                while(k<r):
                    if k != j:
                        fp += df_confusion[k][j]
                    k+=1
                FP.append(fp)
                TN.append(sum - (fp + fn + tp))

    print "Values:",TP,TN,FP,FN
    return TP,TN,FP,FN
    
print numLabels
dist = getDist(numLabels)
data = getData(dist)

print "="*50
print "Original Data---"
print data
print "Original Data---"
print "="*50
split = cross_validation_split(data)
'''
for i in range(10):
    print split[i].count(0)/1000,split[i].count(1)/1000,split[i].count(2)/1000
'''
chance_baseline_prediction = chance_baseline(data)
print "="*50
print "Chance Baseline Prediction Data---"
print chance_baseline_prediction
print "Chance Baseline Prediction Data---"
print "="*50
majority_baseline_prediction = majority_baseline(data)
print "="*50
print "Majority Baseline Prediction Data---"
print majority_baseline_prediction
print "Majority Baseline Prediction Data---"
print "="*50
#print majority_baseline_prediction
#print "\n\n"

truePos1,trueNeg1,falsePos1,falseNeg1 = getConfusionMatVal(data, chance_baseline_prediction)
truePos2,trueNeg2,falsePos2,falseNeg2 = getConfusionMatVal(data, majority_baseline_prediction)

majority = map(int, list(set(majority_baseline_prediction)))[0]
all = list(set(chance_baseline_prediction))

for i in all:
    if i == majority:
        microAvgPrecision = (truePos1[i]+truePos2[0]) / (truePos1[i]+truePos2[0]+falsePos1[i]+falsePos2[0])
        microAvgRecall = (truePos1[i]+truePos2[0]) / (truePos1[i]+truePos2[0]+falseNeg1[i]+falseNeg2[0])
        fMeasure = 2*(microAvgPrecision*microAvgRecall) / (microAvgPrecision+microAvgRecall)
        print "For %d, Micro Average Precision: %f, Recall: %f, FMeasure: %f"%(i,microAvgPrecision,microAvgRecall,fMeasure)
    else:
        precision = (truePos1[i]) / (truePos1[i]+falsePos1[i])
        recall = (truePos1[i]) / (truePos1[i]+falseNeg1[i])
        fmeasure = 2*(precision*recall) / (precision+recall)
        print "For %d, Precision: %f, Recall: %f, FMeasure: %f"%(i,precision,recall,fmeasure)
