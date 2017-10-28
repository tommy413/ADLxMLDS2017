# -*- coding: utf-8 -*-

import sys
import numpy as np
import os
import pickle as pkl

data_path = sys.argv[1]
dim_dict = {"mfcc" : 39, "fbank" : 69}
inputType = sys.argv[2]

def load_map():
	map48to39 = {}
	map39toChar = {}

	map48to39Str = open(data_path + "phones/48_39.map").read()
	map48to39List = [t for t in map48to39Str.split("\n") if t != "" ]
	for row in map48to39List:
		elements = row.split("\t")
		map48to39[elements[0]] = elements[1]

	map39toCharStr = open(data_path + "48phone_char.map").read()
	map39toCharList = [t for t in map39toCharStr.split("\n") if t != ""]
	for row in map39toCharList:
		elements = row.split("\t")
		map39toChar[elements[0]] = elements[2]

	return map48to39,map39toChar

def train_standardize(array):
	means = np.mean(array,axis = 0)
	stds = np.std(array,axis = 0)
	array = (array - means) / stds 
	return array.tolist(),means,stds

def test_standardize(array,means,stds):
	return (array - means) / stds

def to_categorical(tag,tags_list): 
	tags_num = 1
	tags_class = len(tags_list)
	Y_data = np.zeros(tags_class,dtype = 'int')
	Y_data[tag]=1
	return Y_data

def Padding_X(array,maxLength,dim):
	for row in array:
		for i in range(0,maxLength-len(row)):
			row.append(np.zeros(dim,dtype="float64"))
	return array

def Padding_Y(array,maxLength,tag):
	for row in array:
		for i in range(0,maxLength-len(row)):
			row.append(tag)
	return array


def feature_input(typeName):
	input_path = data_path + typeName +"/"
	label_path = data_path + "label/train.lab"

	map48to39,map39toChar = load_map()
	labelFile = open(label_path,'r').read()
	trainFile = open(input_path + "train.ark",'r').read()
	testFile = open(input_path + "test.ark",'r').read()
	print("Files Loaded...")

	trainId2Ix = {}
	ids = []
	Xtrain = []

	trainRows = trainFile.split("\n")
	count = 0
	for row in trainRows :
		elements = row.split(" ")
		if len(elements) != dim_dict[typeName]+1:
			continue
		trainId2Ix[elements[0]] = count
		count = count + 1
		Xtrain.append(elements[1:])
		ids.append(elements[0])

	Xtrain, means, stds = train_standardize(np.array(Xtrain, dtype="float64"))
	Xtrain = [ [ids[i]] + Xtrain[i] for i in range(0,len(ids))]

	label2Ix = {}
	labelList = []
	count = 0

	labelRows = labelFile.split("\n")
	for row in labelRows :
		elements = row.split(",")
		if len(elements) != 2 :
			continue
		label = map39toChar[ map48to39[elements[1]] ]
		labelIx = label2Ix.get(label,-1)
		if labelIx == -1 :
			labelList.append(label)
			label2Ix[label] = count
			labelIx = count
			count = count + 1
		Xtrain[trainId2Ix[elements[0]]].append(labelIx)

	trainId2Ix = {}
	Xtrain_new = []
	Ytrain = []
	sens_X = []
	sens_Y = []
	count = 0
	sentenceId = ""
	maxLength = 0

	for row in Xtrain:
		newId = row[0].split("_")[0] + "_" + row[0].split("_")[1]
		if sentenceId != newId :
			trainId2Ix[newId] = count
			count = count + 1
			sentenceId = newId
			if len(sens_X) > 0 :
				Xtrain_new.append(sens_X)
				Ytrain.append(sens_Y)
				maxLength = max(maxLength,len(sens_X))
				sens_X = []
				sens_Y = []
		else :
			sens_X.append(row[1:-1])
			sens_Y.append(to_categorical(row[-1],labelList))
	
	Xtrain_new.append(sens_X)
	Ytrain.append(sens_Y)
	Xtrain = Xtrain_new

	print("Xtrain,Ytrain finished.")

	testId2Ix = {}
	Xtest = []
	sentenceId = ""
	sens = []
	testRows = testFile.split("\n")
	count = 0

	for row in testRows :
		elements = row.split(" ")
		if len(elements) != dim_dict[typeName]+1:
			continue
		newId = elements[0].split("_")[0] + "_" + elements[0].split("_")[1]
		if sentenceId != newId :
			testId2Ix[newId] = count
			count = count + 1
			sentenceId = newId
			if len(sens) > 0 :
				Xtest.append(sens)
				maxLength = max(maxLength,len(sens))
				sens = []
		else :
			sens.append( test_standardize( np.array(elements[1:], dtype = "float64"), means, stds ) )
	Xtest.append(sens)

	print("Xtest finished.")

	Xtrain = Padding_X(Xtrain,maxLength,dim_dict[typeName])
	Xtest = Padding_X(Xtest,maxLength,dim_dict[typeName])
	Ytrain = Padding_Y(Ytrain,maxLength,to_categorical(label2Ix['L'],labelList))
	print("Padding finished")

	return trainId2Ix,testId2Ix,label2Ix,np.array( Xtrain, dtype = "float64"),np.array(Xtest , dtype = "float64"),np.array(Ytrain, dtype = "int")

trainId2Ix,testId2Ix,label2Ix,Xtrain,Xtest,Ytrain= feature_input(inputType)

outputPath = "feature/%s" % inputType
if not (os.path.exists(outputPath)):
	os.makedirs(outputPath)

pkl.dump(trainId2Ix, open("%s/trainId2Ix.pkl" % outputPath,'wb'), pkl.HIGHEST_PROTOCOL)
pkl.dump(testId2Ix, open("%s/testId2Ix.pkl" % outputPath,'wb'), pkl.HIGHEST_PROTOCOL)
pkl.dump(label2Ix, open("%s/label2Ix.pkl" % outputPath,'wb'), pkl.HIGHEST_PROTOCOL)
pkl.dump(Xtrain, open("%s/Xtrain.pkl" % outputPath,'wb'), pkl.HIGHEST_PROTOCOL)
pkl.dump(Xtest, open("%s/Xtest.pkl" % outputPath,'wb'), pkl.HIGHEST_PROTOCOL)
pkl.dump(Ytrain, open("%s/Ytrain.pkl" % outputPath,'wb'), pkl.HIGHEST_PROTOCOL)

print("All files dumped.")

