# -*- coding: utf-8 -*-

import sys
import numpy as np
import pickle as pkl
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv1D, Conv2D, Reshape, Flatten, Bidirectional, TimeDistributed, GRU
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model,model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
#os.environ["THEANO_FLAGS"] = "device=gpu0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import json

data_path = "feature/mfcc"
model_name = "cnn"
dim_dict = {"mfcc" : 39, "fbank" : 69}

# parameter
nb_epoch = 500
nb_batch = 15
split_ratio = 0.33

Xtrain_raw = pkl.load(open("%s/Xtrain.pkl" % data_path,'rb'))
Ytrain_raw = pkl.load(open("%s/Ytrain.pkl" % data_path,'rb'))


print("Files loaded")

def split_data(X,Y,split_ratio):
	indices = np.arange(X.shape[0])  
	np.random.shuffle(indices) 

	X_data = X[indices]
	Y_data = Y[indices]

	num_validation_sample = int(split_ratio * X_data.shape[0] )

	X_train = X_data[num_validation_sample:]
	Y_train = Y_data[num_validation_sample:]

	X_val = X_data[:num_validation_sample]
	Y_val = Y_data[:num_validation_sample]

	return (X_train,Y_train),(X_val,Y_val)

def check_min_lebal(Y):
	a = Y.reshape(Y.shape[0]*Y.shape[1],Y.shape[2])
	s = np.sum(a,axis = 0)
	ans = s.min()
	print(ans)
	return ans > 2100

def get_model(name):
	
	model = Sequential()
	print('Building model.')
	model.add(Conv1D(800,(39),activation='relu',padding = 'same' ,batch_input_shape = (None,Xtrain.shape[1],Xtrain.shape[2])))
	model.add(Bidirectional(GRU(500,dropout = 0.3,activation='relu', return_sequences=True)))
	model.add(Bidirectional(GRU(500,dropout = 0.3,activation='relu', return_sequences=True)))
	model.add(Bidirectional(GRU(250,dropout = 0.3,activation='relu', return_sequences=True)))
	model.add(Bidirectional(GRU(250,dropout = 0.3,activation='relu', return_sequences=True)))
	model.add(TimeDistributed(Dense(39,activation='softmax')))

	adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])
	with open("model/%s.json" % name, 'w') as f:
		json.dump(model.to_json(), f)

	model.summary()
	earlystopping = EarlyStopping(monitor='val_loss', patience = 7, verbose=1, mode='min')
	checkpoint = ModelCheckpoint(filepath="model/%s_model_weight.hdf5" % name,
	                             verbose=1,
	                             save_best_only=True,
	                             save_weights_only=True,
	                             monitor='val_loss',
	                             mode='min')
	return model,earlystopping,checkpoint


(Xtrain,Ytrain),(Xval,Yval) = split_data(Xtrain_raw,Ytrain_raw,split_ratio)

while not (check_min_lebal(Ytrain)):
	(Xtrain,Ytrain),(Xval,Yval) = split_data(Xtrain_raw,Ytrain_raw,split_ratio)

model,earlystopping,checkpoint = get_model(model_name)
    
hist = model.fit(Xtrain, Ytrain, 
                 validation_data=(Xval, Yval),
                 epochs=nb_epoch, 
                 batch_size=nb_batch,
                 callbacks=[earlystopping,checkpoint])

if not (os.path.exists("history/")):
	os.makedirs(outputPath)("history/")
pkl.dump(hist, open("history/%s.pkl" % model_name,'wb'), pkl.HIGHEST_PROTOCOL)

model.load_weights("model/%s_model_weight.hdf5" % model_name)

score = model.evaluate(Xval, Yval)

print(score)
