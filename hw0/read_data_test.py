import csv
import numpy as np
import sys
import os
os.environ["THEANO_FLAGS"] = "device=gpu0"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

x_train, y_train = load_mnist('fashion-mnist/data/fashion', kind='train')
x_test, y_test = load_mnist('fashion-mnist/data/fashion', kind='t10k')

x_train = x_train.reshape(len(x_train),28,28,1)
x_test = x_test.reshape(len(x_test),28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
x_train = x_train/255
x_test = x_test/255

model = Sequential()
model.add(Conv2D(50,(2,2),input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(100,(2,2)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
# model.add(Conv2D(200,(2,2)))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(400,(2,2)))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=500))
model.add(Dense(units=250,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
batch_size = 200
epoch = 50
model.fit(x_train,y_train,batch_size=batch_size,epochs=epoch)

score = model.evaluate(x_train,y_train)
print('\nLoss:', score[0])
print('\nTrain Acc:', score[1])
score = model.evaluate(x_test,y_test)
print('\nLoss:', score[0])
print('\nTest Acc:', score[1])
model.save("model/CNN_"+str(score[1])+"_"+str(batch_size)+"_"+str(epoch)+".h5")