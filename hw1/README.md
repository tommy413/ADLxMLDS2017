# README : HW1-TIMIT

## Prerequisites

python v3.5.2

需要安裝h5py來存取model
```
pip install h5py
```
## Running the trains
需要先用feature_process.py處理資料

```
python feature_process.py [data_path] mfcc
```

再直接執行py檔即可

```
python model_rnn.py
python model_cnn.py
python model_best.py
```

## Running the test scripts
如sh檔不符權限，請先執行chmod
```
chmod +x xxx.sh
```


