import csv
import os,sys
import random
from feature_process import feature_select,load_pics
import json

from scipy.misc import imsave,imresize

import keras.backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Concatenate, Dropout, BatchNormalization, GaussianNoise, Conv2DTranspose, Multiply, MaxPooling2D, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
import keras.models as kmodels
from keras.utils import to_categorical
from keras.optimizers import Adam

import numpy as np

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# np.random.seed(42)

data_path = os.path.join(sys.argv[1])
model_name = sys.argv[2]
img_size = (64,64)
log = []

# batch and latent size taken from the paper
nb_epochs = 30000
batch_size = 128
emb_size = 8
latent_size = 100

# Adam parameters suggested in https://arxiv.org/abs/1511.06434
adam_lr = 0.0002
adam_beta_1 = 0.5

def make_random_noise(i,nb):
    fake_noise = np.random.normal(0.0,1.0,size = (nb,latent_size))
    hair_noise = np.random.randint(12,size=(nb,1))
    eyes_noise = np.random.randint(11,size=(nb,1))
    fake_hair = to_categorical(hair_noise,num_classes=12)
    fake_eyes = to_categorical(eyes_noise,num_classes=11)

    return fake_noise,hair_noise,eyes_noise,fake_hair,fake_eyes

def build_generator():

    hair_input = Input([1])
    eyes_input = Input([1])
    noise_input = Input([latent_size])

    hair_emb = Flatten()(Embedding(12, emb_size, input_length=1, embeddings_initializer = 'glorot_normal')(hair_input))
    eyes_emb = Flatten()(Embedding(11, emb_size, input_length=1, embeddings_initializer = 'glorot_normal')(eyes_input))

    emb_layer = Concatenate()([hair_emb,eyes_emb])
    emb_layer = Concatenate()([noise_input,emb_layer])

    cnn = Reshape((1,1,-1))(emb_layer)

    cnn = Conv2DTranspose(512, kernel_size = (4, 4), strides = (1,1), padding='valid', kernel_initializer='glorot_normal')(cnn)
    cnn = BatchNormalization(momentum = 0.5)(cnn)
    cnn = LeakyReLU(0.2)(cnn)

    cnn = Conv2DTranspose(256, kernel_size = (4, 4), strides = (2,2), padding='same', kernel_initializer='glorot_normal')(cnn)
    cnn = BatchNormalization(momentum = 0.5)(cnn)
    cnn = LeakyReLU(0.2)(cnn)

    cnn = Conv2DTranspose(128, kernel_size = (4, 4), strides = (2,2), padding='same', kernel_initializer='glorot_normal')(cnn)
    cnn = BatchNormalization(momentum = 0.5)(cnn)
    cnn = LeakyReLU(0.2)(cnn)

    cnn = Conv2DTranspose(64, kernel_size = (4, 4), strides = (2,2), padding='same', kernel_initializer='glorot_normal')(cnn)
    cnn = BatchNormalization(momentum = 0.5)(cnn)
    cnn = LeakyReLU(0.2)(cnn)

    cnn = Conv2D(64, kernel_size = (3, 3), strides = (1,1), padding='same', kernel_initializer='glorot_normal')(cnn)
    cnn = BatchNormalization(momentum = 0.5)(cnn)
    cnn = LeakyReLU(0.2)(cnn)

    cnn_result = Conv2DTranspose(3, kernel_size = (4, 4), strides = (2,2), padding='same', kernel_initializer='glorot_normal',activation='tanh')(cnn)

    generator = kmodels.Model([noise_input, hair_input,eyes_input], cnn_result)
    generator.compile(optimizer=Adam(lr=0.00015, beta_1=0.5),loss='binary_crossentropy')
    generator.summary()
    return generator

def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper

    img_input = Input([64,64,3])

    ### for 96 * 96
    fake_cnn = Conv2D(64, kernel_size = (4, 4), strides = (2,2), padding='same', input_shape=(64, 64, 3), kernel_initializer='glorot_normal')(img_input)
    fake_cnn = LeakyReLU(0.2)(fake_cnn)

    fake_cnn = Conv2D(128, kernel_size = (4, 4), strides = (2,2), padding='same', kernel_initializer='glorot_normal')(fake_cnn)
    fake_cnn = BatchNormalization(momentum = 0.5)(fake_cnn)
    fake_cnn = LeakyReLU(0.2)(fake_cnn)

    fake_cnn = Conv2D(256, kernel_size = (4, 4), strides = (2,2), padding='same', kernel_initializer='glorot_normal')(fake_cnn)
    fake_cnn = BatchNormalization(momentum = 0.5)(fake_cnn)
    fake_cnn = LeakyReLU(0.2)(fake_cnn)

    fake_cnn = Conv2D(512, kernel_size = (4, 4), strides = (2,2), padding='same', kernel_initializer='glorot_normal')(fake_cnn)
    fake_cnn = BatchNormalization(momentum = 0.5)(fake_cnn)
    fake_cnn = LeakyReLU(0.2)(fake_cnn)

    fake_cnn = Flatten()(fake_cnn)
    fake = Dense(1, activation='sigmoid', name='generation')(fake_cnn)
    hair_aux = Dense(12, activation='softmax', name='auxiliary_hair')(fake_cnn)
    eyes_aux = Dense(11, activation='softmax', name='auxiliary_eyes')(fake_cnn)

    discriminator = kmodels.Model(img_input, [fake, hair_aux, eyes_aux])
    discriminator.compile(
    optimizer=Adam(lr=0.0002, beta_1=0.5),
    loss_weights=[1.5,0.75, 0.75],
    loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
    metrics=['accuracy']
    )
    discriminator.summary()

    return discriminator

def combine(g,d):
    hair_input = Input([1])
    eyes_input = Input([1])
    noise_input = Input([latent_size])

    img = g([noise_input, hair_input, eyes_input])
    combine_output = d(img)

    combined_model = kmodels.Model([noise_input, hair_input,eyes_input], combine_output)
    combined_model.compile(
        optimizer=Adam(lr=0.00015, beta_1=0.5),
        loss=['binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
        metrics=['accuracy']
    )
    combined_model.summary()
    return combined_model

def save(g,d,i,log):
    log_file = open(os.path.join("model/%s/" % model_name,"train_log.txt" ),'a')
    for l in log:
        log_file.write("%d,%f,%f,%f,%f,%f,%f,%f,%f,%f" % (l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8],l[9]))
    log = []
    log_file.close()

    g.save_weights(os.path.join("model/%s/" % model_name,"g_%d_model_weight.hdf5" % i ))
    d.save_weights(os.path.join("model/%s/" % model_name,"d_%d_model_weight.hdf5" % i ))

def test_for_train(g,name,epoch):

    output_path = "model_pics/%s/" % name
    if not (os.path.exists(output_path)):
        os.makedirs(output_path)

    for i in range(0,5):
        fake_noise = np.random.normal(0.0,1.0,size = (1,latent_size))

        fake_hair = np.array([[(5 + i) % 12]])
        fake_eyes = np.array([[(7 + i) % 11]])
        # fake_noise,hair_noise,eyes_noise,fake_hair,fake_eyes = make_random_noise(i,1)
        
        pics = g.predict([fake_noise, fake_hair, fake_eyes])
        # pics = g.predict([fake_noise,hair_noise,eyes_noise])
        pics = np.squeeze((pics + 1.0) * 127.5).astype(np.uint8)
        # pics = imresize(pics, (64,64))
        imsave(os.path.join(output_path,"%d_%d.jpg" % (epoch,i) ), pics )

id2tag = feature_select(data_path)
id2pics = load_pics(id2tag.keys(),data_path,img_size)
train_size = len(id2pics.keys())
print("Pics Dataset : %d" % train_size)

# build the discriminator
print('Discriminator model:')
discriminator = build_discriminator()

# build the generator
print('Generator model:')
generator = build_generator()

print('Combined model:')
combined_model = combine(generator, discriminator)


if not (os.path.exists("model/%s/" % model_name)):
    os.makedirs("model/%s/" % model_name)
with open(os.path.join("model/%s/" % model_name,"g.json" ), 'w') as f:
    json.dump(generator.to_json(), f)
with open(os.path.join("model/%s/" % model_name,"d.json" ), 'w') as f:
    json.dump(discriminator.to_json(), f)

discriminator.trainable = False ### important
soft_one = 1.0
soft_zero = 0.0
for i in range(1,nb_epochs+1):
    
    for j in range(1):
        fake_noise,hair_noise,eyes_noise,fake_hair,fake_eyes = make_random_noise(i,batch_size)
        fake_pics = generator.predict([fake_noise, hair_noise, eyes_noise])
        discriminator.trainable = True
        generator.trainable = False

        #sample true pics
        batch_ids = random.sample(list(id2tag.keys()),batch_size)
        true_pics = np.vstack([id2pics[ix] for ix in batch_ids])
        true_hair = np.vstack([id2tag[ix][0] for ix in batch_ids])
        true_eyes = np.vstack([id2tag[ix][1] for ix in batch_ids])
        
        #train for D
        Y_discriminator = [np.ones((batch_size,1)) - np.random.random_sample((batch_size,1))*0.2, np.random.random_sample((batch_size,1))*0.2]
        true_d_loss = discriminator.train_on_batch(true_pics,[Y_discriminator[0],true_hair,true_eyes])
        fake_d_loss = discriminator.train_on_batch(fake_pics,[Y_discriminator[1],fake_hair,fake_eyes])
    
    #train for G
    generator.trainable = True
    discriminator.trainable = False
    for j in range(1):
        fake_noise,hair_noise,eyes_noise,fake_hair,fake_eyes = make_random_noise(i,batch_size)
        g_loss = combined_model.train_on_batch([fake_noise, hair_noise, eyes_noise], [Y_discriminator[0],fake_hair,fake_eyes])
    print("Epoch : %d - true_d_loss : %f,%f,%f - fake_d_loss : %f,%f,%f - g_loss : %f,%f,%f" % (i,true_d_loss[1],true_d_loss[5],true_d_loss[6],fake_d_loss[1],fake_d_loss[5],fake_d_loss[6],g_loss[1],g_loss[5],g_loss[6]))
    log.append([i,true_d_loss[1],true_d_loss[5],true_d_loss[6],fake_d_loss[1],fake_d_loss[5],fake_d_loss[6],g_loss[1],g_loss[5],g_loss[6]])
    if i > 0 and i % 2000 == 0 :
        save(generator,discriminator,i,log)
        log = []
    if i > 0 and i % 100 == 0 :
        test_for_train(generator,model_name,i)


