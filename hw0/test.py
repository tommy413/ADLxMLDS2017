import time
import csv
import numpy as np
import os
import sys
os.environ["THEANO_FLAGS"] = "device=gpu0"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import load_model
#categorical_crossentropy

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    # labels_path = os.path.join(path,
    #                            '%s-labels-idx1-ubyte.gz'
    #                            % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    # with gzip.open(labels_path, 'rb') as lbpath:
    #     labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
    #                            offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(10000, 784)

    return images

x_test= load_mnist('', kind='t10k')

x_test = x_test.reshape(len(x_test),28,28,1)
x_test = x_test.astype('float32')

x_test = x_test/255

# print(y_test)

model_name = "model/model1.h5"
model = load_model(model_name)
result = model.predict(x_test)

# result_file = open("result/result_"+model_name+".csv","w")
result_file = open("rst.csv","w")
rw = csv.writer(result_file)
rw.writerow(["id","label"])
count = 0
for item in result:
	ma = max(item)
	index = -1
	for i in range(0,len(item)):
		if item[i] == ma :
			index = i
	rw.writerow([count,index])
	count = count + 1
result_file.close()