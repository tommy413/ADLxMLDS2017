# -*- coding: utf-8 -*-

import sys
import numpy as np
import pickle as pkl
import os
import csv
from keras.models import load_model,model_from_json
#os.environ["THEANO_FLAGS"] = "device=gpu0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json

data_path = "feature/%s" % sys.argv[1]
model_name = ["best_1","best_2","best_3"]
output_path = sys.argv[2]
dim_dict = {"mfcc" : 39, "fbank" : 69}

testId2Ix = pkl.load(open("%s/testId2Ix.pkl" % data_path,'rb'))
label2Ix = pkl.load(open("%s/label2Ix.pkl" % data_path,'rb'))
Xtest = pkl.load(open("%s/Xtest.pkl" % data_path,'rb'))
print("Files loaded.")

def average(model_1,model_2,model_3,Ix2label):
	result_1 = model_1.predict(Xtest)
	result_2 = model_2.predict(Xtest)
	result_3 = model_3.predict(Xtest)
	#result_4 = model_3.predict(Xtest)

	result = []
	for i in range(0,Xtest.shape[0]):
		row = []
		for j in range(0,Xtest.shape[1]):
			rst = np.array([ result_1[i][j] ,result_2[i][j] ,result_3[i][j] ], dtype = "float64")
			means = np.mean(rst, axis = 0 )
			row.append(np.argmax(means))
		result.append(row)

	return result

def voting(model_1,model_2,model_3,Ix2label):
	result_1 = model_1.predict_classes(Xtest)
	result_2 = model_2.predict_classes(Xtest)
	result_3 = model_3.predict_classes(Xtest)

	result = []
	for i in range(0,Xtest.shape[0]):
		row = []
		for j in range(0,Xtest.shape[1]):
			max_dict = {}
			max_dict[result_1[i][j]] = max_dict.get(result_1[i][j],0) + 1
			max_dict[result_2[i][j]] = max_dict.get(result_2[i][j],0) + 1
			max_dict[result_3[i][j]] = max_dict.get(result_3[i][j],0) + 1

			max_v = 0
			max_l = ""
			for l in max_dict.keys():
				if max_dict[l] > max_v:
					max_v = max_dict[l]
					max_l = l
			row.append(max_l)
		result.append(row)
	return result

def trimming(seq):
	flag = 0
	last = ""
	count = 0
	new_seq = ""

	for i in seq:
		if i != last :
			last = i
			if flag == 0 :
				if i != 'L' :
					if count >= 2 :
						new_seq = new_seq + i
						flag = 1
			elif flag == 1 :
				if count >= 2 :
					if new_seq[-1] != i:
						new_seq = new_seq + i
			count = 1
		elif i == last :
			count = count + 1
	
	while new_seq[-1] == 'L' :
		new_seq = new_seq[:-1]

	return new_seq

Ix2label = {}
for key in label2Ix.keys():
	Ix2label[label2Ix[key]] = key

model_1 = model_from_json(json.load(open("model/%s.json" % model_name[0])))
model_1.load_weights("model/%s_model_weight.hdf5" % model_name[0])
model_1.summary()
print('Model_1 loaded.')

model_2 = model_from_json(json.load(open("model/%s.json" % model_name[1])))
model_2.load_weights("model/%s_model_weight.hdf5" % model_name[1])
model_2.summary()
print('Model_2 loaded.')

model_3 = model_from_json(json.load(open("model/%s.json" % model_name[2])))
model_3.load_weights("model/%s_model_weight.hdf5" % model_name[2])
model_3.summary()
print('Model_3 loaded.')

# model_4 = model_from_json(json.load(open("model/%s.json" % model_name[3])))
# model_4.load_weights("model/%s_model_weight.hdf5" % model_name[3])
# model_4.summary()
# print('Model_4 loaded.')

result = voting(model_1,model_2,model_3,Ix2label)
#result = average(model_1,model_2,model_3,Ix2label)

Ix2TestId = {}
for key in testId2Ix.keys():
	Ix2TestId[testId2Ix[key]] = key

final_rst = []
for row in result:
	rstStr = ""
	for element in row :
		rstStr = rstStr + str(Ix2label[element])
	final_rst.append(rstStr)

trimed_rst = []
for row in final_rst:
	trimed_rst.append( trimming(list(row)) )

result_file = open(output_path,'w')
wf = csv.writer(result_file)
wf.writerow(['id','phone_sequence'])
for i in range(0,len(trimed_rst)):
	wf.writerow([Ix2TestId[i],trimed_rst[i]])
result_file.close()
