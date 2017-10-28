# -*- coding: utf-8 -*-

import sys
import numpy as np
import pickle as pkl
import os
import csv
#os.environ["THEANO_FLAGS"] = "device=gpu0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from keras.models import load_model,model_from_json
import json

data_path = "feature/%s" % sys.argv[1]
model_name = sys.argv[2]
output_path = sys.argv[3]
dim_dict = {"mfcc" : 39, "fbank" : 69}

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
					if count >= 3 :
						new_seq = new_seq + i
						flag = 1
			elif flag == 1 :
				if count >= 3 :
					if new_seq[-1] != i:
						new_seq = new_seq + i
			count = 1
		elif i == last :
			count = count + 1
	
	while new_seq[-1] == 'L' :
		new_seq = new_seq[:-1]

	return new_seq

testId2Ix = pkl.load(open("%s/testId2Ix.pkl" % data_path,'rb'))
label2Ix = pkl.load(open("%s/label2Ix.pkl" % data_path,'rb'))
Xtest = pkl.load(open("%s/Xtest.pkl" % data_path,'rb'))
print("Files loaded.")

Ix2label = {}
for key in label2Ix.keys():
	Ix2label[label2Ix[key]] = key

model = model_from_json(json.load(open("model/%s.json" % model_name)))
model.load_weights("model/%s_model_weight.hdf5" % model_name)
model.summary()
print('Model loaded.')

result = model.predict_classes(Xtest)

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
