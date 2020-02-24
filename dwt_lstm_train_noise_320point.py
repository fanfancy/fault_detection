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
   
from readdata_dwt import load_data_320
from dwt_model import *
from dwtprocess import *
  
####read file ############
# linenum_all = ['49', '269', '313', '316', '75', '72', '69', '66']
linenum_all = ['316']
# linenum_all = ['100002']
writefile = 'train_data'
suffix = '_320point.txt'

SNR = [40,35,30]
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

'''
read the filename
'''
for linenum in linenum_all:
    curdir = os.path.abspath(os.curdir)
    filename = os.path.join(curdir,'./data/'+linenum+writefile+suffix)
    length = 320
    diff_conditions = 5
    kinds = 360*diff_conditions
    
    X_train, y_train = load_data_320(filename,length,kinds)
    y_train = y_train[0::length]
    X_train = np.reshape(X_train,[-1,length,3])

    index = [i for i in range(X_train.shape[0])]
    np.random.shuffle(index)
    data_ori = X_train[index,:,:]
    label = y_train[index,:]
    
    for snr in SNR:
        
        print('SNR is '+str(snr))
        data_noise = np.zeros([kinds, length, 3])
        for i in tqdm(range(kinds)):
            for j in range(3):
                data_noise[i,:,j] = wgn(data_ori[i,:,j], snr)

        data = data_ori + data_noise

        # input_dwt,dwt_length = dwtprocesswhole(data1,kinds)

        res = multicore(data, kinds)
        dwt_length = res[0][1]
        print (dwt_length)
        input_dwt = np.zeros((kinds, dwt_length))
        for i in range(kinds):
            input_dwt[i, :] = res[i][0]

        global_start_time = time.time()
        epochs = 800
        layers = [3, length, 128, 32, 4, dwt_length]
        model_type = build_model_type(layers)

        # model_type.summary()
        # plot_model(model_type, to_file='model.png')

        h_type = model_type.fit(
            {'main_input': data, 'dwt_input':input_dwt},
            {'output': (label[:, 0:4])},
            batch_size=int(kinds*0.75),
            validation_split=0.25,
            verbose=0,
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
#         with open("log2.txt",'a+') as f:
#             f.write("@ train %s.\n" % (linenum))
#             f.write("@ train fault types.\n")
#             f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[-1] * 100, m_acc + 1))
#             f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[-1] * 100, m_val_acc + 1))
        # K.clear_session()
        # epochs = 2000
        global_start_time = time.time()
        layers = [3, length, 128, 32, 4, dwt_length]
        model_phase = build_model_phase(layers)
        
        h_phase = model_phase.fit(
            {'main_input':data,'dwt_input':input_dwt},
            {'output':(label[:,4:8])},
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
        print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." \
              % (acc[m_acc] * 100, m_acc + 1))
        print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." \
              % (val_acc[m_val_acc] * 100, m_val_acc + 1))
        print("@ Last Training Accuracy: %.2f %%." \
              % (acc[-1] * 100))
        print("@ Last Testing Accuracy: %.2f %%." % \
              (val_acc[-1] * 100))
#         with open("log2.txt", 'a+') as f:
#             f.write("@ train %s.\n" % (linenum))
#             f.write("@ train fault phases.\n")
#             f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
#             f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
#         K.clear_session()
        
        layers = [3, length, 128, 32, 1, dwt_length]
#         epochs = 2000
        model_loc = build_model_location(layers)
        global_start_time = time.time()
        h_loc = model_loc.fit(
            {'main_input':data,'dwt_input':input_dwt},
            {'output':(label[:,-1])},
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
        print("@ Best Training mae: %.2f %% achieved at EP #%d." % \
              (mae[m_mae] * 100, m_mae + 1))
        print("@ Best Testing mae: %.2f %% achieved at EP #%d." % \
              (val_mae[m_val_mae] * 100, m_val_mae + 1))
        print("@ Last Training Accuracy: %.2f %%." \
              % (acc[-1] * 100))
        print("@ Last Testing Accuracy: %.2f %%." % \
              (val_acc[-1] * 100))
#         with open("log2.txt", 'a+') as f:
#             f.write("@ train %s.\n" % (linenum))
#             f.write("@ train fault locations.\n")
#             f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_mae] * 100, m_mae + 1))
#             f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_mae[m_val_mae] * 100, m_val_mae + 1))
#         K.clear_session()

        '''
        global_start_time = time.time()

        epochs = 2000
        model_type = build_model_type([3, length, 128, 32, 4],dwt_length)

        h_type = model_type.fit(
                {'main_input':data1,'dwt_input':input_dwt},
                {'output':(label1[:,0:4])},
            #     {'output':(label2[:,8])},
                batch_size=720,
                validation_split=0.3,
                verbose=0,
                shuffle=True,
                epochs=epochs)

        print('Training duration (s) : ', time.time() - global_start_time)
        acc = h_type.history['acc']
        val_acc = h_type.history['val_acc']
        m_acc = np.argmax(acc)
        m_val_acc = np.argmax(val_acc)
        print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
        print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
        with open("log.txt",'w+') as f:
            f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
            f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
        '''



# for linenum in linenum_all:
#     print(linenum," trainging test")
#     curdir = os.path.abspath(os.curdir)
#     filename = os.path.join(curdir,'./data/'+linenum+writefile+suffix)


#     data_ori, label = load_data(filename,length,kinds)
#     label = label[0::length]
#     data_ori = np.reshape(data_ori,[-1,length,3])
    
#     for snr in SNR:
        
#         print('SNR is '+str(snr))
#         data_noise = np.zeros([kinds, length, 3])
#         for i in tqdm(range(kinds)):
#             for j in range(3):
#                 data_noise[i,:,j] = wgn(data_ori[i,:,j], snr)

#         data = data_ori + data_noise

#         #################### whether using HHT and its coe or not ###############
#         if net_mode == 1:
#             ### data for CNN only
#             pass
#         # else:
#         res = multicore(data, n_imfs, kinds)
#         data2 = np.zeros((kinds, length, n_imfs * 2, 3))
#         for i in range(kinds):
#             data2[i, :, :, :] = res[i]

#             # if net_mode == 3 or net_mode ==4:
#         start = int(length * 0.25)
#         end = -start
#         HHT_coefficients = np.zeros((kinds,6,2*n_imfs,3))
#         import scipy.stats

#         for kind in range(kinds):
#             for i in range(2*n_imfs):
#                 for j in range(3):
#                     HHT_coefficients[kind,0,i,j] = np.max(          data2[kind,start:end,i,j])
#                     HHT_coefficients[kind,1,i,j] = np.min(          data2[kind,start:end,i,j])
#                     HHT_coefficients[kind,2,i,j] = np.mean(         data2[kind,start:end,i,j])
#                     HHT_coefficients[kind,3,i,j] = np.std(          data2[kind,start:end,i,j])
#                     HHT_coefficients[kind,4,i,j] = scipy.stats.skew(   data2[kind,start:end,i,j])
#                     HHT_coefficients[kind,5,i,j] = np.sum(np.square(   data2[kind,start:end,i,j]))


#         HHT_coefficients_ori = np.reshape(HHT_coefficients,[kinds,-1])

#         ################### adjust the format to input directly into CNN##############
#         data3 = data[:,:,:,np.newaxis]

    
#         #############shuffle the data #######################################
#         index = [i for i in range(data.shape[0])]
#         np.random.shuffle(index)
#         data2_type = data2[index,:,:,:]
#         data3_type = data3[index,:,:,:]
#         HHT_coefficients_type = HHT_coefficients_ori[index,:]
#         label_type = label[index,:]
#         print (label_type.shape)
#         ################ choose the network structure ##############
#         if net_mode==1:
#             input_data = data3_type
#         elif net_mode==2:
#             input_data = data2_type
#         elif net_mode==3:
#             input_data = {'main_input':data2_type,'hht_co_input':HHT_coefficients_type}
#         elif net_mode==4:
#             input_data = {'main_input':data3_type,'hht_co_input':HHT_coefficients_type}
#         elif net_mode==5:
#             input_data = {'main_input':data2_type,'hht_co_input':HHT_coefficients_type}

#         ##################Type test ################################1
#         input_label = label_type[:,0:5]
#         global_start_time = time.time()
#         model_type = build_model_type(layers)
#         h_type = model_type.fit(
#                     input_data,
#                     input_label,
#                     batch_size=int(kinds*0.75),
#                     validation_split=0.25,
#                     verbose=1,
#                     shuffle=True,
#                     epochs=epochs)
#         print('Type Training duration (s) : ', time.time() - global_start_time)
#         acc = h_type.history['acc']
#         val_acc = h_type.history['val_acc']
#         m_acc = np.argmax(acc)
#         m_val_acc = np.argmax(val_acc)
#         print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
#         print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
#         print("@ Last Training Accuracy: %.2f %% ." % (acc[-1] * 100))
#         print("@ Last Testing Accuracy: %.2f %% ." % (val_acc[-1] * 100))

#         if record:
#             with open("log_htt2.txt",'a+') as f:
#                 f.write("@ train %s.\n" % (linenum))
#                 f.write("@ train fault types.\n")
#                 f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
#                 f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
#         K.clear_session()

# #         ############## Phase test ########################2

# #         #############shuffle the data #######################################
# #         index_phase = [i for i in range(fault_kinds)]
# #         np.random.shuffle(index_phase)
# #         data2_phase = data2[index_phase,:,:,:]
# #         data3_phase = data3[index_phase,:,:,:]
# #         HHT_coefficients_phase = HHT_coefficients_ori[index_phase,:]
# #         label_phase = label[index_phase,:]
# #         print (label_phase.shape)


# #         if net_mode==1:
# #             input_data = data3_phase
# #         elif net_mode==2:
# #             input_data = data2_phase
# #         elif net_mode==3:
# #             input_data = {'main_input':data2_phase,'hht_co_input':HHT_coefficients_phase}
# #         elif net_mode==4:
# #             input_data = {'main_input':data3_phase,'hht_co_input':HHT_coefficients_phase}
# #         elif net_mode==5:
# #             input_data = {'main_input':data2_phase,'hht_co_input':HHT_coefficients_phase}

# #         model_phase = build_model_phase(layers)
# #         input_label = label_phase[:,4:8]
# #         global_start_time = time.time()
# #         h_phase = model_phase.fit(
# #                 input_data,
# #                 input_label,
# #                 batch_size=int(kinds*0.75),
# #                 validation_split=0.25,
# #                 verbose=0,
# #                 shuffle=True,
# #                 epochs=epochs)

# #         print('Phase Training duration (s) : ', time.time() - global_start_time)
# #         acc = h_phase.history['acc']
# #         val_acc = h_phase.history['val_acc']
# #         print("@ Last Training Accuracy: %.2f %% ." % (acc[-1] * 100))
# #         print("@ Last Testing Accuracy: %.2f %% ." % (val_acc[-1] * 100))
# #         m_acc = np.argmax(acc)
# #         m_val_acc = np.argmax(val_acc)
# #         print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
# #         print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
# #         if record:
# #             with open("log_htt2.txt", 'a+') as f:
# #                 f.write("@ train %s.\n" % (linenum))
# #                 f.write("@ train fault phases.\n")
# #                 f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
# #                 f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
# #         K.clear_session()

# #         ############ Location test #######################3
# #         model_loc = build_model_location(layers)
# #         input_label = label_phase[: ,-1]
# #         global_start_time = time.time()
# #         h_loc = model_loc.fit(
# #                 input_data,
# #                 input_label,
# #                 batch_size=int(kinds*0.75),
# #                 validation_split=0.25,
# #                 verbose=0,
# #                 shuffle=True,
# #                 epochs=epochs)
# #         print('Location Training duration (s) : ', time.time() - global_start_time)
# #         mae = h_loc.history['mean_absolute_error']
# #         val_mae = h_loc.history['val_mean_absolute_error']
# #         m_mae = np.argmin(mae)
# #         m_val_mae = np.argmin(val_mae)
# #         print("@ Best Training mae: %.2f %% achieved at EP #%d." % (mae[m_mae] * 100, m_mae + 1))
# #         print("@ Best Testing mae: %.2f %% achieved at EP #%d." % (val_mae[m_val_mae] * 100, m_val_mae + 1))
# #         print("@ Last Training Accuracy: %.2f %% ." % (mae[-1] * 100))
# #         print("@ Last Testing Accuracy: %.2f %% ." % (val_mae[-1] * 100))
# #         if record:
# #             with open("log_htt2.txt", 'a+') as f:
# #                 f.write("@ train %s.\n" % (linenum))
# #                 f.write("@ train fault locations.\n")
# #                 f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_mae] * 100, m_mae + 1))
# #                 f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_mae[m_val_mae] * 100, m_val_mae + 1))
# #         K.clear_session()
