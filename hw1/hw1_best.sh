pip install h5py
python feature_process.py "$1" mfcc
python test_ensemble.py mfcc "$2"