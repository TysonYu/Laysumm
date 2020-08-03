# data preprocess
use preprocess/preprocess.ipynb

# make data loader
python preprocess.py -data_name=abs_intro -mode=train
python preprocess.py -data_name=abs_intro -mode=valid

# train basic model
python train.py -visible_gpu=3 -data_name=abs_intro -epoch=1000

# inference model
python inference_laysumm.py -data_name=abs_intro -train_from=./save/abs_intro/**.chkpt
