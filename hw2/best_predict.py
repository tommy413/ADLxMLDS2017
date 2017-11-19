# -*- coding: utf-8 -*-

import sys
import numpy as np
import pickle as pkl
import os,re,string
import csv

import keras.models as kmodels
import keras.layers
from keras.optimizers import SGD, Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import keras.backend as K
from keras.models import load_model,model_from_json
import json

data_path = sys.argv[1]
model_name = "best"
test_output_path = sys.argv[2]
peer_output_path = sys.argv[3]
nb_words = 2000
max_len = 50
latent_dim = 1024

tokenizer = pkl.load(open("feature/tokenizer_%d.pkl" % nb_words,'rb'))
word_index = tokenizer.word_index
word_index[''] = 0

id2word = {}
for k in word_index.keys():
	id2word[word_index[k]] = k

def data_input(dtype):
    feat = {}
    train_path = data_path + "training_data/"
    test_path = data_path + "testing_data/"
    peer_path = data_path + "peer_review/"
    feat_path = ""
    
    if dtype == "train" :
        feat_path = train_path + "feat/"
    elif dtype == "test" :
        feat_path = test_path + "feat/"
    elif dtype == "peer" :
        feat_path = peer_path + "feat/"
    
    for path in os.listdir(feat_path):
        feat[path.replace(".npy","")] = np.load(feat_path + path)

    return feat

def make_sen(sen):
	bos_flag = -1
	#eos_flag = -1
	max_len_sen = []
	ans = ""

	for i in range(0,len(sen)):
		if sen[i] == "eos" or sen[i] == "bos" :
			if len(max_len_sen) < (i - bos_flag - 1):
				max_len_sen = sen[bos_flag +1 :i]
				#print(max_len_sen)
			bos_flag = i

	for t in max_len_sen:
		ans = ans + t + " "

	return ans.strip()

def decoder_str(model,Xtest):
	Xtest = Xtest.reshape(1,80,4096)
	Ydata = np.zeros((1,max_len,nb_words + 1))
	Ydata[0,0,word_index['bos']] = 1
	ans = ['bos']

	while len(ans) < max_len and ans[-1] != 'eos':
		y_pred = model.predict([Xtest,Ydata])
		Ydata[0,len(ans),np.argmax(y_pred[0,len(ans)-1,:])] = 1
		ans.append(id2word[np.argmax(y_pred[0,len(ans)-1,:])])
	
	return ans

test_dict = data_input("test")
peer_dict = data_input("peer")

print("Files loaded.")

video_input = keras.layers.Input([80,4096])
decoder_input = keras.layers.Input(shape=(max_len, nb_words + 1))

encoder_output, state_h, state_c = keras.layers.LSTM(latent_dim, return_state=True, return_sequences=True)(video_input)
encoder_state = [state_h, state_c]

hidden = keras.layers.Permute((2,1))(encoder_output)
hidden = keras.layers.Dense(max_len, activation='relu')(hidden)
hidden = keras.layers.Permute((2,1))(hidden)

attention = keras.layers.Dense(latent_dim, activation='softmax')(hidden)

hidden = keras.layers.Concatenate()([hidden, attention])
hidden = keras.layers.Dense(latent_dim, activation='relu')(hidden)

video_decoder = keras.layers.LSTM(latent_dim,activation='relu',return_sequences=True)(hidden)

decoder_output, _, _ = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)(decoder_input,initial_state=encoder_state)

decoder_dense = keras.layers.Concatenate()([video_decoder,decoder_output])

result = keras.layers.Dense(nb_words + 1, activation='softmax')(decoder_dense)

model = kmodels.Model([video_input, decoder_input], result)


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights("model/%s_model_weight.hdf5" % model_name)
model.summary()
print('Model loaded.')

test_result_file = open(test_output_path,'w')
for ix in sorted(list(test_dict.keys())):
	ans = decoder_str(model,np.array(test_dict[ix]))
	test_result_file.write("%s,%s\n" % (ix,make_sen(ans)))
test_result_file.close()

peer_result_file = open(peer_output_path,'w')
for ix in sorted(list(peer_dict.keys())):
	ans = decoder_str(model,np.array(peer_dict[ix]))
	peer_result_file.write("%s,%s\n" % (ix,make_sen(ans)))
peer_result_file.close()
