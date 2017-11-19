wget https://www.dropbox.com/s/1ovokjr0vzlqlzs/best_model_weight.hdf5?dl=1 -O model/best_model_weight.hdf5 
pip install h5py
python best_predict.py "$1" "$2" "$3"