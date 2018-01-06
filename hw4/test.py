import csv
import os,sys
import random
from feature_process import feature_select,load_pics
import json
import pickle

from scipy.misc import imsave

from keras.models import load_model,model_from_json
from keras.utils.vis_utils import plot_model
import numpy as np

hair_encode_dict = {'orange hair' : 0, 'white hair': 1, 'aqua hair': 2, 'gray hair': 3,
    'green hair': 4, 'red hair': 5, 'purple hair': 6, 'pink hair': 7,
    'blue hair': 8, 'black hair': 9, 'brown hair': 10, 'blonde hair': 11}
eyes_encode_dict = {
    'gray eyes': 0, 'black eyes': 1, 'orange eyes': 2,
    'pink eyes': 3, 'yellow eyes': 4, 'aqua eyes': 5, 'purple eyes': 6,
    'green eyes': 7, 'brown eyes': 8, 'red eyes': 9, 'blue eyes': 10}

# data_path = os.path.join(sys.argv[1])
test_path = os.path.join(sys.argv[1])
# model_name = sys.argv[2]
img_size = (64,64)

def load_model():
    g = model_from_json(json.load(open(os.path.join("model/" ,"g.json" ))))
    plot_model(g,to_file=os.path.join("model","g.png"),show_shapes = True)
    g.load_weights(os.path.join("model/","g_model_weight.hdf5" ))

    d = model_from_json(json.load(open(os.path.join("model/" ,"d.json" ))))
    plot_model(d,to_file=os.path.join("model","d.png"),show_shapes = True)
    # d.load_weights(os.path.join("model/","d_model_weight.hdf5" ))

    return g,d

def load_test():
    test_dict = dict()
    test_file = open(test_path,'r')
    test_rd = csv.reader(test_file)
    label = []
    for line in test_rd:
        ix = int(line[0])
        strs = line[1].split(" ")
        for i in range(1,len(strs),2):
            label.append("%s %s" % (strs[i-1],strs[i]))
        hair_id = 9
        eyes_id = 10
        for s in label:
            if s in hair_encode_dict.keys():
                hair_id = hair_encode_dict[s]
            if s in eyes_encode_dict.keys():
                eyes_id = eyes_encode_dict[s]
        test_dict[ix] = [hair_id, eyes_id]

    return test_dict


if __name__ == "__main__":
    seed = 7122
    np.random.seed(seed)
    # seed_file = open("samples/%d.txt" % seed,'w')

    g, d = load_model()
    test_dict = load_test()
    ix_list = sorted(test_dict.keys())

    for ix in ix_list:
        for j in range(1,6):
            noise = np.random.normal(0.0,1.0,size = (1,100))
            hair = np.array(test_dict[ix][0]).reshape(1,1)
            eyes = np.array(test_dict[ix][1]).reshape(1,1)
            pics = g.predict([noise, hair, eyes])
            pics = (pics + 1.0) * 127.5
            imsave(os.path.join("samples","sample_%d_%d.jpg" % (ix,j) ), np.squeeze(pics ))
            # seed_file.write("%s - %s\n" % ("sample_%d_%d.jpg" % (ix,j), str(noise)))
            # pkl.dump(noise, os.path.join("samples","sample_%d_%d.pkl" % (ix,j) ), pkl.HIGHEST_PROTOCOL)
    

