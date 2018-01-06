import csv
import os,sys
import string,re
import numpy as np

import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Concatenate, Dropout, BatchNormalization, GaussianNoise, Conv2DTranspose, Multiply, MaxPooling2D, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
import keras.models as kmodels
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import json

hair_encode_dict = {'orange hair' : 0, 'white hair': 1, 'aqua hair': 2, 'gray hair': 3,
    'green hair': 4, 'red hair': 5, 'purple hair': 6, 'pink hair': 7,
    'blue hair': 8, 'black hair': 9, 'brown hair': 10, 'blonde hair': 11}
eyes_encode_dict = {
    'gray eyes': 0, 'black eyes': 1, 'orange eyes': 2,
    'pink eyes': 3, 'yellow eyes': 4, 'aqua eyes': 5, 'purple eyes': 6,
    'green eyes': 7, 'brown eyes': 8, 'red eyes': 9, 'blue eyes': 10}

# ignore_list = [39,56,70,124,129,256,336,343,353,401,402,416,423,461,517,576,594,599,614,663]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def feature_select(path):
    id2tag = dict()
    tag2id = dict()
    no_eyes = 0
    no_hair = 0

    tag_csv = csv.reader(open(os.path.join(path,"tags_clean.csv"), 'r'))
    for ix in tag_csv:
        img_id = int(ix[0])
        tags = [ s.split(":")[0] for s in ix[1].split("\t") ]
        # select_tags = [t for t in tags if (t in hair_encode_dict.keys() or t in eyes_encode_dict.keys())]
        # hair_onehot = np.zeros((1,len(hair_encode_dict.keys() )) )
        # eyes_onehot = np.zeros((1,len(eyes_encode_dict.keys() )) )
        # hair_onehot = np.random.random_sample((1,12))*0.1
        # eyes_onehot = np.random.random_sample((1,11))*0.1
        # hair_size = 0
        # eyes_size = 0
        # if (len(select_tags) > 0):
        #     for t in select_tags:
        #         hair_id = hair_encode_dict.get(t,-1)
        #         eyes_id = eyes_encode_dict.get(t,-1)
        #         if (hair_id < 0 and eyes_id < 0):
        #             continue

        #         if hair_id >= 0:
        #             hair_size += 1
        #             hair_onehot[0,hair_id] = 1
        #         if eyes_id >= 0:
        #             eyes_size += 1
        #             eyes_onehot[0,eyes_id] = 1
            
        #         tmp = tag2id.get(t,[])
        #         tmp.append(img_id)
        #         tag2id[t] = tmp
        #     if hair_size == 1 and eyes_size == 1:
        #         # if (hair_size == 0):
        #         #     hair_onehot[0] = softmax( np.random.random_sample((1,12)))
        #         #     no_hair += 1
        #         # if (eyes_size == 0):
        #         #     eyes_onehot[0] = softmax( np.random.random_sample((1,11)))
        #         #     no_eyes += 1
                
        #         id2tag[img_id] = [hair_onehot, eyes_onehot]
                # print(id2tag[img_id])
        hair_id = [hair_encode_dict[t] for t in tags if t in hair_encode_dict.keys()]
        eyes_id = [eyes_encode_dict[t] for t in tags if t in eyes_encode_dict.keys()]
        if len(hair_id) == 1 and len(eyes_id) == 1:
            hair_onehot = to_categorical(hair_id,num_classes=12)
            eyes_onehot = to_categorical(eyes_id,num_classes=11)

            id2tag[img_id] = [hair_onehot,eyes_onehot]

    #print(no_hair,no_eyes)
    return id2tag

def load_pics(id_set,path,size):
    import scipy.misc

    pics_path = os.path.join(path,"faces")
    pics_dict = dict()

    for ix in id_set:
        img = scipy.misc.imread(os.path.join(pics_path,"%d.jpg" % ix) , mode = "RGB" )
        #print(img.shape)
        img_resized = scipy.misc.imresize(img, size)
        img_resized = img_resized / 127.5 - 1.0
        #scipy.misc.imsave(os.path.join("pics_test","%d.jpg" %  ix), img_resized)
        #print(img_resized.shape)
        pics_dict[ix] = np.expand_dims(img_resized,axis = 0)
        scipy.misc.imsave(os.path.join("pics_test","%d.jpg" %  ix), np.squeeze((img_resized + 1.0) * 127.5))

    return pics_dict

def get_hair_model(name):
    if os.path.isfile("model/%s_hair.json" % name):
        return None,None,None

    model = kmodels.Sequential()

    model.add(Conv2D(64, kernel_size = (4, 4),strides=(2,2), padding='valid', input_shape=(64, 64, 3)))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.25))
    # model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, kernel_size = (2, 2), padding='valid'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.25))
    # model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, kernel_size = (2, 2), padding='valid'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # model.add(Conv2D(256, kernel_size = (2, 2), padding='valid'))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # # model.add(BatchNormalization(momentum = 0.5))
    # # model.add(LeakyReLU(0.2))

    # model.add(Conv2D(512, kernel_size = (2, 2), padding='valid'))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(BatchNormalization(momentum = 0.5))
    # model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(units=512,activation='relu'))
    model.add(Dense(units=256,activation='relu'))
    model.add(Dense(units=12,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

    with open("model/%s_hair.json" % name, 'w') as f:
        json.dump(model.to_json(), f)
    model.summary()
    earlystopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath="model/%s_hair_model_weight.hdf5" % name,
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min')
    return model,earlystopping,checkpoint

def get_eyes_model(name):
    if os.path.isfile("model/%s_eyes.json" % name):
        return None,None,None

    model = kmodels.Sequential()

    model.add(Conv2D(64, kernel_size = (4, 4),strides=(2,2), padding='valid', input_shape=(64, 64, 3)))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.25))
    # model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, kernel_size = (2, 2), padding='valid'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.25))
    # model.add(BatchNormalization(momentum = 0.5))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, kernel_size = (2, 2), padding='valid'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LeakyReLU(0.2))

    # model.add(Conv2D(256, kernel_size = (2, 2), padding='valid'))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # # model.add(BatchNormalization(momentum = 0.5))
    # # model.add(LeakyReLU(0.2))

    # model.add(Conv2D(512, kernel_size = (2, 2), padding='valid'))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(BatchNormalization(momentum = 0.5))
    # model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(units=512,activation='relu'))
    model.add(Dense(units=256,activation='relu'))
    model.add(Dense(units=11,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

    with open("model/%s_eyes.json" % name, 'w') as f:
        json.dump(model.to_json(), f)
    model.summary()
    earlystopping = EarlyStopping(monitor='val_loss', patience = 5, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath="model/%s_eyes_model_weight.hdf5" % name,
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 mode='min')
    return model,earlystopping,checkpoint

def split_data(X_data,Y_data,split_ratio):
    indices = np.arange(X_data.shape[0])  
    np.random.shuffle(indices) 

    # X_data = X_data[indices]
    # Y_data = Y_data[indices]

    num_validation_sample = int(split_ratio * X_data.shape[0] )

    ix_train = indices[num_validation_sample:]
    ix_val = indices[:num_validation_sample]

    # X_train = X_data[num_validation_sample:]
    # Y_train = Y_data[num_validation_sample:]

    # X_val = X_data[:num_validation_sample]
    # Y_val = Y_data[:num_validation_sample]

    return (X_data[ix_train],Y_data[ix_train]),(X_data[ix_val],Y_data[ix_val])

if __name__ == "__main__":
    data_path = os.path.join(sys.argv[1])
    pretrain_flag = int(sys.argv[2]) > 0
    id2tag,tag2id = feature_select(data_path)
    print(len(id2tag.items()))

    if pretrain_flag :
        model_name = "test_7"

        size_int = int(sys.argv[3])
        nb_epoch = 500
        nb_batch = 200

        id_list = sorted(list(id2tag.keys()))
        id2pics = load_pics(id2tag.keys(),data_path,(size_int,size_int))
        X = []
        y_hair = []
        y_eyes = []
        for i in id_list:
            X.append(id2pics[i])
            y_hair.append(id2tag[i][0])
            y_eyes.append(id2tag[i][1])

        X = np.vstack(X)
        y_hair = np.vstack(y_hair)
        y_eyes = np.vstack(y_eyes)

        hair_model,hair_earlystopping,hair_checkpoint = get_hair_model(model_name)
        eyes_model,eyes_earlystopping,eyes_checkpoint = get_eyes_model(model_name)

        if hair_model != None :
            (Xtrain, Ytrain), (Xval, Yval) = split_data(X,y_hair,0.01)
            hair_model.fit(Xtrain, Ytrain, 
                     validation_data=(Xval, Yval),
                     epochs=nb_epoch, 
                     batch_size=nb_batch,
                     callbacks=[hair_earlystopping,hair_checkpoint])
        if eyes_model != None :
            (Xtrain, Ytrain), (Xval, Yval) = split_data(X,y_eyes,0.01)
            eyes_model.fit(Xtrain, Ytrain, 
                     validation_data=(Xval, Yval),
                     epochs=nb_epoch, 
                     batch_size=nb_batch,
                     callbacks=[eyes_earlystopping,eyes_checkpoint])