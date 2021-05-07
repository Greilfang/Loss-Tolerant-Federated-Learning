import datetime
import os
import h5py
import numpy as np
# https://www.cnblogs.com/-wenli/p/14020264.html
# f = h5py.File('path/filename.h5','r') #打开h5文件
f = h5py.File('Mnist_pFedMe_0.005_1.0_15_5u_20b_10_5_0.09_avg.h5','r')
print(f.filename, ":")
print([key for key in f.keys()], "\n")  

#['rs_glob_acc', 'rs_train_acc', 'rs_train_loss'] 

d = f['rs_glob_acc']

# Print the data of 'dset'.
print(d.name, ":")
print(type(d[:]))
print('-----------------------------------')
print(len(d.attrs.keys()))
# Print the attributes of dataset 'dset'.
for key in d.attrs.keys():
    print(key, ":", d.attrs[key])
