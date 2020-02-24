import time
import numpy as np
from numpy import random as nr
import os
from tqdm import tqdm


import keras
from keras.models import Model
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras import backend as K
# from keras.callbacks import Reduc

from readdata import load_data

from httprocess import *


#####work mode####
CNN_only = 1
if CNN_only:
    from cnn_model import *
else:
    from htt_coe_cnn_model import *

####read file ############
#linenum_all = ['313']
linenum_all = ['49']
# linenum_all = ['49', '269', '313', '316', '75', '72', '69', '66']
# linenum_all = ['100002']
writefile = 'train_data'
#suffix = '_320point.txt'
suffix = '_320point_large_resistance.txt'

#dataformat parameters
length = 400
diff_conditions = 4
repeat_times = 10
normal = diff_conditions * repeat_times
kinds = 540*diff_conditions + normal
n_imfs = 2
#training parameters
epochs = 10000
if CNN_only:
    layers = [length,3,1]
else:
    layers = [length,2*n_imfs,3]

'''
read the filename
'''


for linenum in linenum_all:
    print(linenum," trainging test")
    curdir = os.path.abspath(os.curdir)
    filename = os.path.join(curdir,'./data/'+linenum+writefile+suffix)


    X_train, y_train = load_data(filename,length,kinds)
    y_train = y_train[0::length]
    X_train = np.reshape(X_train,[-1,length,3])

    index = [i for i in range(X_train.shape[0])]
    np.random.shuffle(index)
    data1 = X_train[index,:,:]
    label1 = y_train[index,:]
    print (data1.shape)
    
    
    
    if CNN_only:
        ### data for CNN only
        data3 = data1[:,:,:,np.newaxis]
    else:
        
        res = multicore(data1, n_imfs, kinds)
        data2 = np.zeros((kinds, length, n_imfs * 2, 3))
        for i in range(kinds):
            data2[i, :, :, :] = res[i]


        HHT_coefficients = np.zeros((kinds,6,2*n_imfs,3))
        import scipy.stats

        for kind in range(kinds):
            for i in range(2*n_imfs):
                for j in range(3):
                    HHT_coefficients[kind,0,i,j] = np.max(          data2[kind,:,i,j])
                    HHT_coefficients[kind,1,i,j] = np.min(          data2[kind,:,i,j])
                    HHT_coefficients[kind,2,i,j] = np.mean(         data2[kind,:,i,j])
                    HHT_coefficients[kind,3,i,j] = np.std(          data2[kind,:,i,j])
                    HHT_coefficients[kind,4,i,j] = scipy.stats.skew(data2[kind,:,i,j])
                    HHT_coefficients[kind,5,i,j] = np.sum(np.square(data2[kind,:,i,j]))


        HHT_coefficients = np.reshape(HHT_coefficients,[kinds,-1])
    
    
    
    global_start_time = time.time()
    model_type = build_model_type(layers)
    
    if CNN_only:
        h_type = model_type.fit(
                data3,
                label1[:, 0:5],
                batch_size=int(kinds*0.25),
                validation_split=0.25,
                verbose=1,
                shuffle=True,
                epochs=epochs)
    else:
        h_type = model_type.fit(
                {'main_input':data2,'hht_co_input':HHT_coefficients},
                {'output':label1[:,0:5]},
                batch_size=int(kinds*0.25),
                validation_split=0.25,
                verbose=1,
                shuffle=True,
                epochs=epochs)

    print('Type Training duration (s) : ', time.time() - global_start_time)
    acc = h_type.history['acc']
    val_acc = h_type.history['val_acc']
    m_acc = np.argmax(acc)
    m_val_acc = np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    print("@ Last Training Accuracy: %.2f %% ." % (acc[-1] * 100))
    print("@ Last Testing Accuracy: %.2f %% ." % (val_acc[-1] * 100))
#     with open("log_htt2.txt",'a+') as f:
#         f.write("@ train %s.\n" % (linenum))
#         f.write("@ train fault types.\n")
#         f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
#         f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    K.clear_session()

    
    
    model_phase = build_model_phase(layers)
    global_start_time = time.time()
    if CNN_only:
        h_phase = model_phase.fit(
            data3,
            label1[:, 5:9],
            batch_size=int(kinds*0.7),
            validation_split=0.3,
            verbose=0,
            shuffle=True,
            epochs=epochs)
    else:
        h_phase = model_phase.fit(
                {'main_input':data2,'hht_co_input':HHT_coefficients},
                {'output':label1[:,5:9]},
                batch_size=int(kinds*0.25),
                validation_split=0.25,
                verbose=1,
                shuffle=True,
                epochs=epochs)

    print('Phase Training duration (s) : ', time.time() - global_start_time)
    acc = h_phase.history['acc']
    val_acc = h_phase.history['val_acc']
    print("@ Last Training Accuracy: %.2f %% ." % (acc[-1] * 100))
    print("@ Last Testing Accuracy: %.2f %% ." % (val_acc[-1] * 100))
    m_acc = np.argmax(acc)
    m_val_acc = np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
# #     with open("log_htt2.txt", 'a+') as f:
# #         f.write("@ train %s.\n" % (linenum))
# #         f.write("@ train fault phases.\n")
# #         f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
# #         f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    K.clear_session()

    model_loc = build_model_location(layers)
    global_start_time = time.time()
    
    if CNN_only:
        h_loc = model_loc.fit(
            data3,
            label1[:, -1],
            batch_size=int(kinds*0.7),
            validation_split=0.3,
            verbose=0,
            shuffle=True,
            epochs=epochs)
    else:
        h_loc = model_loc.fit(
                {'main_input':data2,'hht_co_input':HHT_coefficients},
                {'output':label1[:,-1]},
                batch_size=int(kinds*0.25),
                validation_split=0.25,
                verbose=1,
                shuffle=True,
                epochs=epochs)

    print('Location Training duration (s) : ', time.time() - global_start_time)
    mae = h_loc.history['mean_absolute_error']
    val_mae = h_loc.history['val_mean_absolute_error']
    m_mae = np.argmin(mae)
    m_val_mae = np.argmin(val_mae)
    print("@ Best Training mae: %.2f %% achieved at EP #%d." % (mae[m_mae] * 100, m_mae + 1))
    print("@ Best Testing mae: %.2f %% achieved at EP #%d." % (val_mae[m_val_mae] * 100, m_val_mae + 1))
    print("@ Last Training Accuracy: %.2f %% ." % (mae[-1] * 100))
    print("@ Last Testing Accuracy: %.2f %% ." % (val_mae[-1] * 100))
# #     with open("log_htt2.txt", 'a+') as f:
# #         f.write("@ train %s.\n" % (linenum))
# #         f.write("@ train fault locations.\n")
# #         f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_mae] * 100, m_mae + 1))
# #         f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_mae[m_val_mae] * 100, m_val_mae + 1))
    K.clear_session()
