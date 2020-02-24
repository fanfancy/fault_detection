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
from htt_model import *
from httprocess import *


# linenum_all = ['49', '269', '313', '316', '75', '72', '69', '66']
linenum_all = ['316']
# linenum_all = ['100002']
writefile = 'train_data'
suffix = '_320point.txt'
'''
read the filename
'''
# lr = 0.001
# def scheduler(epoch,lr=lr):
#     if epoch %1000==0 and epoch!=0:
#         # lr = K.get_value(model.optimizer.lr)
#         K.setvalue(model.optimizer.lr, lr*0.1)
#         lr = lr*0.1
#     return lr

for linenum in linenum_all:
    curdir = os.path.abspath(os.curdir)
    filename = os.path.join(curdir,'./data/'+linenum+writefile+suffix)

    length = 320
    diff_conditions = 5
    kinds = 360*diff_conditions

    X_train, y_train = load_data(filename,length,kinds)
    y_train = y_train[0::length]
    X_train = np.reshape(X_train,[-1,length,3])

    index = [i for i in range(X_train.shape[0])]
    np.random.shuffle(index)
    data1 = X_train[index,:,:]
    label1 = y_train[index,:]

    n_imfs = 2
    res = multicore(data1, n_imfs, kinds)
    data2 = np.zeros((kinds, 320, n_imfs * 2, 3))
    for i in range(kinds):
        data2[i, :, :, :] = res[i]


    global_start_time = time.time()

    epochs = 3000
    layers = [320,4]
    model_type = build_model_type(layers)

    

    # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=10,mode="auto")
    # reduce_lr = keras.callbacks.LearningRateScheduler(scheduler)
    h_type = model_type.fit(
            data2,
            label1[:, 0:4],
            batch_size=int(kinds*0.75),
            validation_split=0.25,
            verbose=1,
            shuffle=True,
            epochs=epochs)

    print('Training duration (s) : ', time.time() - global_start_time)
    acc = h_type.history['acc']
    val_acc = h_type.history['val_acc']
    m_acc = np.argmax(acc)
    m_val_acc = np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    print("@ Last Training Accuracy: %.2f %% ." % (acc[-1] * 100))
    print("@ Last Testing Accuracy: %.2f %% ." % (val_acc[-1] * 100))
    with open("log_htt2.txt",'a+') as f:
        f.write("@ train %s.\n" % (linenum))
        f.write("@ train fault types.\n")
        f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
        f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    K.clear_session()

    model_phase = build_model_phase(layers)
    global_start_time = time.time()
    h_phase = model_phase.fit(
        data2,
        label1[:, 4:8],
        batch_size=int(kinds*0.75),
        validation_split=0.25,
        verbose=0,
        shuffle=True,
        epochs=epochs)

    print('Training duration (s) : ', time.time() - global_start_time)
    acc = h_phase.history['acc']
    val_acc = h_phase.history['val_acc']
    m_acc = np.argmax(acc)
    m_val_acc = np.argmax(val_acc)
    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    with open("log_htt2.txt", 'a+') as f:
        f.write("@ train %s.\n" % (linenum))
        f.write("@ train fault phases.\n")
        f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
        f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
    K.clear_session()

    epochs = 3000
    model_loc = build_model_location(layers)
    global_start_time = time.time()
    h_loc = model_loc.fit(
        data2,
        label1[:, -1],
        batch_size=int(kinds*0.75),
        validation_split=0.25,
        verbose=0,
        shuffle=True,
        epochs=epochs)

    print('Training duration (s) : ', time.time() - global_start_time)
    mae = h_loc.history['mean_absolute_error']
    val_mae = h_loc.history['val_mean_absolute_error']
    m_mae = np.argmin(mae)
    m_val_mae = np.argmin(val_mae)
    print("@ Best Training mae: %.2f %% achieved at EP #%d." % (mae[m_mae] * 100, m_mae + 1))
    print("@ Best Testing mae: %.2f %% achieved at EP #%d." % (val_mae[m_val_mae] * 100, m_val_mae + 1))
    with open("log_htt2.txt", 'a+') as f:
        f.write("@ train %s.\n" % (linenum))
        f.write("@ train fault locations.\n")
        f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_mae] * 100, m_mae + 1))
        f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_mae[m_val_mae] * 100, m_val_mae + 1))
    K.clear_session()
