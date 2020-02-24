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

from httprocess import *
   
from readdata_dwt import load_data_400
from dwt_model import *
from dwtprocess import *
  
####read file ############
# linenum_all = ['49', '269', '313', '316', '75', '72', '69', '66']
linenum_all = ['49']
# linenum_all = ['316']
# linenum_all = ['100002']
writefile = 'train_data'
# suffix = '_400point_large_resistance.txt'
suffix = '_400point_3load_multime_large.txt'

'''
read the filename
'''
for linenum in linenum_all:
    curdir = os.path.abspath(os.curdir)
    filename = os.path.join(curdir,'./data/'+linenum+writefile+suffix)
    
    length = 400
    diff_conditions = 8                       #load change nums
    loc = 9                               #loaction types
    fault_type = 10                          #AG,BG,..,ABCG fault types
    resistance = 5                          # resistance types [0.01, 0.1, 1, 10, 100 ...]
    n_imfs = 2                             # nimf nums
    repeat_times = resistance * loc
    normal = diff_conditions * repeat_times
    fault_kinds = loc*fault_type*resistance*diff_conditions
    kinds = fault_kinds + normal
    
    X_train, y_train = load_data_400(filename,length,kinds)
    y_train = y_train[0::length]
    X_train = np.reshape(X_train,[-1,length,3])

    index = [i for i in range(X_train.shape[0])]
    np.random.shuffle(index)
    data_ori = X_train[index,:,:]
    label = y_train[index,:]
    
    data = data_ori

    # input_dwt,dwt_length = dwtprocesswhole(data1,kinds)

    res = multicore(data, kinds)
    dwt_length = res[0][1]
    print (dwt_length)
    input_dwt = np.zeros((kinds, dwt_length))
    for i in range(kinds):
        input_dwt[i, :] = res[i][0]

    global_start_time = time.time()
    epochs = 800
    layers = [3, length, 128, 32, 5, dwt_length]
    model_type = build_model_type(layers)

    # model_type.summary()
    # plot_model(model_type, to_file='model.png')

    h_type = model_type.fit(
        {'main_input': data, 'dwt_input':input_dwt},
        {'output': (label[:, 0:5])},
        batch_size=int(kinds*0.25),
        validation_split=0.25,
        verbose=1,
        shuffle=True,
        epochs=epochs)

    print('Training duration (s) : ', time.time() - global_start_time)
    acc = h_type.history['acc']
    val_acc = h_type.history['val_acc']
    m_acc = np.argmax(acc)
    m_val_acc = np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." \
          % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." \
          % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    print("@ Last Training Accuracy: %.2f %%." \
          % (acc[-1] * 100))
    print("@ Last Testing Accuracy: %.2f %%." % \
          (val_acc[-1] * 100))
    with open("log2.txt",'a+') as f:
        f.write("@ train %s.\n" % (linenum))
        f.write("@ train fault types.\n")
        f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
        f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
        f.write("@ Last Training Accuracy: %.2f %%.\n" % (acc[-1] * 100))
        f.write("@ Last Testing Accuracy: %.2f %%.\n" % (val_acc[-1] * 100))
    K.clear_session()
    epochs = 2000
    global_start_time = time.time()
    layers = [3, length, 128, 32, 4, dwt_length]
    model_phase = build_model_phase(layers)
    
    h_phase = model_phase.fit(
        {'main_input':data,'dwt_input':input_dwt},
        {'output':(label[:,5:9])},
        batch_size=int(kinds*0.25),
        validation_split=0.25,
        verbose=1,
        shuffle=True,
        epochs=epochs)
    
    print('Training duration (s) : ', time.time() - global_start_time)
    acc = h_phase.history['acc']
    val_acc = h_phase.history['val_acc']
    m_acc = np.argmax(acc)
    m_val_acc = np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." \
          % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." \
          % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    print("@ Last Training Accuracy: %.2f %%." \
          % (acc[-1] * 100))
    print("@ Last Testing Accuracy: %.2f %%." % \
          (val_acc[-1] * 100))
    with open("log2.txt", 'a+') as f:
        f.write("@ train %s.\n" % (linenum))
        f.write("@ train fault phases.\n")
        f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
        f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
        f.write("@ Last Training Accuracy: %.2f %%.\n" % (acc[-1] * 100))
        f.write("@ Last Testing Accuracy: %.2f %%.\n" % (val_acc[-1] * 100))
    K.clear_session()
    
    layers = [3, length, 128, 32, 1, dwt_length]
    epochs = 2000
    model_loc = build_model_location(layers)
    global_start_time = time.time()
    h_loc = model_loc.fit(
        {'main_input':data,'dwt_input':input_dwt},
        {'output':(label[:,-1])},
        batch_size=int(kinds*0.25),
        validation_split=0.25,
        verbose=1,
        shuffle=True,
        epochs=epochs)
    
    print('Training duration (s) : ', time.time() - global_start_time)
    mae = h_loc.history['mean_absolute_error']
    val_mae = h_loc.history['val_mean_absolute_error']
    m_mae = np.argmin(mae)
    m_val_mae = np.argmin(val_mae)
    print("@ Best Training mae: %.2f %% achieved at EP #%d." % \
          (mae[m_mae] * 100, m_mae + 1))
    print("@ Best Testing mae: %.2f %% achieved at EP #%d." % \
          (val_mae[m_val_mae] * 100, m_val_mae + 1))
    print("@ Last Training Accuracy: %.2f %%." \
          % (mae[-1] * 100))
    print("@ Last Testing Accuracy: %.2f %%." % \
          (val_mae[-1] * 100))
    with open("log2.txt", 'a+') as f:
        f.write("@ train %s.\n" % (linenum))
        f.write("@ train fault locations.\n")
        f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_mae] * 100, m_mae + 1))
        f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_mae[m_val_mae] * 100, m_val_mae + 1))
        f.write("@ Last Training Accuracy: %.2f %%.\n" % (mae[-1] * 100))
        f.write("@ Last Testing Accuracy: %.2f %%.\n" % (val_mae[-1] * 100))

    K.clear_session()
