
# coding: utf-8
#from multiprocessing import Pool

import os,sys
import re, string
import numpy as np
import json
import pickle as pkl

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import keras.models as kmodels
import keras.layers
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model,model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint

data_path = sys.argv[1]
model_name = "best"

#parameters
split_ratio = 0.2
nb_epoch = 500
nb_batch = 100
nb_words = 2000
max_len = 50

def data_input(dtype):
    feat = {}
    train_path = data_path + "training_data/"
    test_path = data_path + "testing_data/"
    feat_path = ""
    
    if dtype == "train" :
        feat_path = train_path + "feat/"
    elif dtype == "test" :
        feat_path = test_path + "feat/"
    
    for path in os.listdir(feat_path):
        feat[path.replace(".npy","")] = np.load(feat_path + path)

    return feat

def label_input(dtype):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    if dtype == "train" :
        labels_f = json.load(open(data_path + "training_label.json"))
    elif dtype == "test" :
        labels_f = json.load(open(data_path + "testing_label.json"))
    
    labels = {}
    for row in labels_f :
        labels[row['id']] = ["<bos> %s <eos>" % s for s in row['caption']]

    return labels

def feature_input():

    train_dict = data_input("train")

    train_labels = label_input("train")
    train_words = []

    print("Load Files")

    if not (os.path.exists("feature/")):
        os.makedirs("feature/")

    if not os.path.isfile("feature/tokenizer_%d.pkl" % nb_words) :
        for t in train_labels.values():
            for s in t:
                train_words.append(s)

        tokenizer = Tokenizer(num_words = nb_words)
        tokenizer.fit_on_texts(train_words)
        pkl.dump(tokenizer, open("feature/tokenizer_%d.pkl" % nb_words ,'wb'), pkl.HIGHEST_PROTOCOL)
    else :
        tokenizer = pkl.load(open("feature/tokenizer_%d.pkl" % nb_words,'rb'))

    word_index = tokenizer.word_index
    word_index[''] = 0

    print("Build word index")

    if not (os.path.exists("model/")):
        os.makedirs("model/")
    
    print(len(word_index))

    Xdata = []
    Ydata = []

    for ix in train_labels.keys():
        train_sequences = tokenizer.texts_to_sequences(train_labels[ix])
        train_sequences = [to_categorical(t, num_classes=nb_words + 1) for t in train_sequences]

        indices = np.arange(len(train_sequences))  
        np.random.shuffle(indices)
        ixs = indices[0:min(len(train_sequences),5)]

        for s in np.array(train_sequences)[ixs] :
            Xdata.append(train_dict[ix])
            Ydata.append(s)

    for i in range(0,len(Ydata)):
        Ydata[i] = np.lib.pad(Ydata[i],((0,max_len - Ydata[i].shape[0]),(0,0)), 'edge')
    
    print("Finish padding")

    Xdata = np.array(Xdata, dtype = "float16")
    Ydata = np.array(Ydata)

    print("Get feature")

    return Xdata,Ydata

def split_data(X_data,Y_data,split_ratio):
    indices = np.arange(X_data.shape[0])  
    np.random.shuffle(indices) 

    num_validation_sample = int(split_ratio * X_data.shape[0] )

    ix_train = indices[num_validation_sample:]
    ix_val = indices[:num_validation_sample]

    return (X_data[ix_train],Y_data[ix_train]),(X_data[ix_val],Y_data[ix_val])

def get_model(name,max_len,n_class):
    latent_dim = 1024

    if not (os.path.exists("model/")):
        os.makedirs("model/")
    
    print('Model Building.')

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
    with open("model/%s.json" % model_name, 'w') as f:
        json.dump(model.to_json(), f)

    model.summary()
    earlystopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath="model/%s_model_weight.hdf5" % model_name,
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min')
    return model,earlystopping,checkpoint

#Train
Xtrain,Ytrain = feature_input()
(Xtrain,Ytrain),(Xval,Yval) = split_data(Xtrain,Ytrain,split_ratio)

Ytarget = Ytrain[:,1:,:]
Ytargetval = Yval[:,1:,:]
Ytarget = np.lib.pad(Ytarget,((0,0),(0,max_len - Ytarget.shape[1]),(0,0)), 'edge')
Ytargetval = np.lib.pad(Ytargetval,((0,0),(0,max_len - Ytargetval.shape[1]),(0,0)), 'edge')

model,earlystopping,checkpoint = get_model(model_name,max_len,nb_words+1)

hist = model.fit([Xtrain, Ytrain],Ytarget, 
                 validation_data=([Xval, Yval],Ytargetval),
                 epochs=nb_epoch, 
                 batch_size=nb_batch,
                 callbacks=[earlystopping,checkpoint])


if not (os.path.exists("history/")):
    os.makedirs("history/")

with open('history/%s' % (model_name), 'wb') as file_pi:
        pkl.dump(hist.history, file_pi)

model.load_weights("model/%s_model_weight.hdf5" % model_name)

score = model.evaluate([Xval, Yval],Ytargetval)

print(score)

del Xtrain,Ytrain,Xval,Yval,model,earlystopping,checkpoint

outFile = open("model/scoreList.txt",'a')
outFile.write("%s\t%s\t%s\n" % (model_name,str(score[0]),str(score[1])))