pip install h5py
python feature_process.py "$1" mfcc
python test.py mfcc cnn "$2"