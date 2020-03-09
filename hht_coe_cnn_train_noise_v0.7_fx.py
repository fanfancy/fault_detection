import time
import numpy as np
from numpy import random as nr
import os
from tqdm import tqdm
import sys

import keras
from keras.models import Model
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
# from keras.callbacks import Reduc

from readdata import load_data

from httprocess2 import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt    


####read file ############
linenum_all = ['313']
#linenum_all = ['49']
# linenum_all = ['49', '269', '313', '316', '75', '72', '69', '66']
# linenum_all = ['100002']
writefile = 'train_data'
# suffix = '_320point.txt'
suffix = '_1600point_randtime300.txt'



### whhether save the result or not
record = 1
record_file = "0229_noise_mode6.txt"
#dataformat parameters
length = 1600                            #length of the seuqence
diff_conditions = 1                       #load change nums
loc = 9                               #loaction types
fault_type = 11                          #AG,BG,..,ABCG fault types
resistance = 4                          # resistance types [0.01, 0.1, 1, 10, 100 ...]
n_imfs = 2                             # nimf nums
repeat_times = 300
normal = diff_conditions
fault_kinds = fault_type
kinds = (fault_kinds + normal)*repeat_times
net_mode = 6

SNR = [40, 35, 30]
def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

#training parameters
epochs = 3000
faulttype = 1
faultphase = 0
faultloc = 0
#####work mode################################################
'''
net_mode = 1  using current data directly
net_mode = 2  using HHT processed data without HHT coe
net_mode = 3  using HHT processed data with HHT coe
net_mode = 4  using current data directly with HHT coe
net_mode = 5  cnn kernel test
'''


if net_mode == 1:
    from cnn_model_new import *
    layers = [length,3,1,n_imfs]
elif net_mode == 2 :
    from cnn_model_new import *
    layers = [length,2*n_imfs,3,n_imfs]
elif net_mode == 3 :
    from htt_coe_cnn_model_new import *
    layers = [length,2*n_imfs,3,n_imfs]
elif net_mode == 4:
    from htt_coe_cnn_model_new import *
    layers = [length,3,1,n_imfs]
elif net_mode == 5:
    from cnn_kernel_test_model import *
    layers = [length,2*n_imfs,3,n_imfs]
elif net_mode == 6:                        
    from cnn_model_2channel import *
    layers = [length,3,2]
#######################################################


for linenum in linenum_all:
    print(linenum," trainging test")
    curdir = os.path.abspath(os.curdir)
    filename = os.path.join(curdir,'./data/'+linenum+writefile+suffix)

    data_ori, label, name = load_data(filename,length,kinds,fault_type)
    label = label[0::length]
    data_ori = np.reshape(data_ori,[-1,length,3])

    test_size = 0.25
    train_size = 1 - test_size
    seed = 7
    data_train, data_test, label_train, label_test = train_test_split(data_ori, label, test_size=test_size, random_state=seed)

    for snr in SNR:
        print('SNR is '+str(snr))
        data_train_noise = np.zeros([int(kinds*train_size), length, 3])
        for i in tqdm(range(int(kinds*train_size))):
            snr_train = SNR[i%3]
            # snr_train = snr
            for j in range(3):
                data_train_noise[i,:,j] = wgn(data_train[i,:,j], snr)

        data_train = data_train + data_train_noise


        data_test_noise = np.zeros([int(kinds*test_size), length, 3])
        for i in tqdm(range(int(kinds*test_size))):
            for j in range(3):
                data_test_noise[i,:,j] = wgn(data_test[i,:,j], snr)
        data_test = data_test + data_test_noise 

        #################### whether using HHT and its coe or not ###############
        if net_mode == 1:
            ### data for CNN only
            pass
        # else:            
        res_train = multicore(data_train, 3, int(kinds*train_size))
        res_train = np.array(res_train)
        res_test = multicore(data_test, 3, int(kinds*test_size))
        res_test = np.array(res_test)
        print (res_train.shape)
        print (res_test.shape)
        data2 = np.zeros((int(kinds*train_size), length, n_imfs * 2, 3))
        for i in range(int(kinds*train_size)):
            data2[i, :, :, :] = res_train[i, :, 2:, :]
        data4 = np.zeros((int(kinds*test_size), length, n_imfs * 2, 3))
        for i in range(int(kinds*test_size)):
            data4[i, :, :, :] = res_test[i, :, 2:, :]

            # if net_mode == 3 or net_mode ==4:
        start = int(length * 0.25)
        end = -start
        HHT_coefficients_train = np.zeros((int(kinds*train_size),6,n_imfs,3))
        HHT_coefficients_test = np.zeros((int(kinds*test_size),6,n_imfs,3))
        import scipy.stats

        for kind in range(int(kinds*train_size)):
            for i in range(n_imfs):
                for j in range(3):
                    HHT_coefficients_train[kind,0,i,j] = np.max(          data2[kind,start:end,2*i+1,j])
                    HHT_coefficients_train[kind,1,i,j] = np.min(          data2[kind,start:end,2*i+1,j])
                    HHT_coefficients_train[kind,2,i,j] = np.mean(         data2[kind,start:end,2*i+1,j])
                    HHT_coefficients_train[kind,3,i,j] = np.std(          data2[kind,start:end,2*i+1,j])
                    HHT_coefficients_train[kind,4,i,j] = scipy.stats.skew(   data2[kind,start:end,2*i+1,j])
                    HHT_coefficients_train[kind,5,i,j] = np.sum(np.square(   data2[kind,start:end,2*i+1,j]))

        for kind in range(int(kinds*test_size)):
            for i in range(n_imfs):
                for j in range(3):
                    HHT_coefficients_test[kind,0,i,j] = np.max(          data4[kind,start:end,2*i+1,j])
                    HHT_coefficients_test[kind,1,i,j] = np.min(          data4[kind,start:end,2*i+1,j])
                    HHT_coefficients_test[kind,2,i,j] = np.mean(         data4[kind,start:end,2*i+1,j])
                    HHT_coefficients_test[kind,3,i,j] = np.std(          data4[kind,start:end,2*i+1,j])
                    HHT_coefficients_test[kind,4,i,j] = scipy.stats.skew(   data4[kind,start:end,2*i+1,j])
                    HHT_coefficients_test[kind,5,i,j] = np.sum(np.square(   data4[kind,start:end,2*i+1,j]))

        HHT_coefficients_ori_train = np.reshape(HHT_coefficients_train,[int(kinds*train_size),-1])
        HHT_coefficients_ori_test = np.reshape(HHT_coefficients_test,[int(kinds*test_size),-1])

        ################### adjust the format to input directly into CNN##############
        data3 = data_train[:,:,:,np.newaxis]
        data5 = data_test[:,:,:,np.newaxis]
    
        #############shuffle the data #######################################
        index = [i for i in range(data_train.shape[0])]
        np.random.shuffle(index)
        data2_type = data2[index,:,:,:]
        data3_type = data3[index,:,:,:]
        HHT_coefficients_type_train = HHT_coefficients_ori_train[index,:]
        label_type_train = label_train[index,:]
        print (label_type_train.shape)

        index = [i for i in range(data_test.shape[0])]
        data4_type = data4[index,:,:,:]
        data5_type = data5[index,:,:,:]
        HHT_coefficients_type_test =  HHT_coefficients_ori_test[index,:]
        label_type_test = label_test[index,:]
        print (label_type_test.shape)

        print ("data3_type",data3_type.shape)
        
        data2_type_reshape = data2_type[:,:,1,:].reshape(int(kinds*train_size),length,3,1)
        print ("data2_type_reshape",data2_type_reshape.shape)
        data_2channel = np.concatenate((data3_type,data2_type_reshape),axis=3)
        print (data_2channel.shape)

        data4_type_reshape = data4_type[:,:,1,:].reshape(int(kinds*test_size),length,3,1)
        print ("data4_type_reshape",data4_type_reshape.shape)
        data_2channel_test = np.concatenate((data5_type,data4_type_reshape),axis=3)
        print (data_2channel_test.shape)

        ################ choose the network structure ##############
        if net_mode==1:
            input_data_train = data3_type
            input_data_test = data5_type
        elif net_mode==2:
            input_data_train = data2_type
            input_data_test = data4_type
        elif net_mode==3:
            input_data_train = {'main_input':data2_type,'hht_co_input':HHT_coefficients_type_train}
            input_data_test = {'main_input':data4_type,'hht_co_input':HHT_coefficients_type_test}
        elif net_mode==4:
            input_data_train = {'main_input':data3_type,'hht_co_input':HHT_coefficients_type_train}
            input_data_test = {'main_input':data5_type,'hht_co_input':HHT_coefficients_type_test}
        elif net_mode==5:
            input_data_train = {'main_input':data2_type,'hht_co_input':HHT_coefficients_type_train}
            input_data_test = {'main_input':data4_type,'hht_co_input':HHT_coefficients_type_test}
        elif net_mode==6:
            input_data_train = data_2channel
            input_data_test = data_2channel_test
        ##################Type test ################################
        if faulttype == 1:
            epochs = 2000
            input_label_train = label_type_train[:,0:6]
            input_label_test = label_type_test[:,0:6]
            global_start_time = time.time()
            model_type = build_model_type1(layers, fault_type)
            h_type = model_type.fit(
                        input_data_train,
                        input_label_train,
                        batch_size=int(kinds*0.3),
                        validation_data=(input_data_test, input_label_test),
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

            if record:
                plt.figure()
                plt.plot(acc)
                plt.savefig("./figure_res/acc_type.png")

                plt.figure()
                plt.plot(val_acc)
                plt.savefig("./figure_res/val_acc_type.png")

                plt.figure()
                loss = h_type.history['loss']
                plt.plot(loss)
                plt.savefig("./figure_res/loss_type.png")

                with open(record_file,'a+') as f:
                    f.write("@ train %s.\n" % (linenum))
                    f.write("@ train fault types.\n")
                    f.write("@ snr = %s.\n" % (snr))
                    f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
                    f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
                    f.write("@ Now Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_val_acc] * 100, m_val_acc + 1))
                    f.write("@ Last Training Accuracy: %.2f %% .\n" % (acc[-1] * 100))
                    f.write("@ Last Testing Accuracy: %.2f %% \n." % (val_acc[-1] * 100))

            K.clear_session()

        ############## Phase test ########################2
        if faultphase == 1:
            epochs = 3000
            model_phase = build_model_phase1(layers)
            input_label_train = label_type_train[:,6:10]
            input_label_test = label_type_test[:,6:10]
            global_start_time = time.time()
            h_phase = model_phase.fit(
                    input_data_train,
                    input_label_train,
                    batch_size=int(kinds*0.3),
                    validation_data=(input_data_test, input_label_test),
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
            if record:
                plt.figure()
                plt.plot(acc)
                plt.savefig("./figure_res/acc_phase.png")

                plt.figure()
                plt.plot(val_acc)
                plt.savefig("./figure_res/val_acc_phase.png")

                plt.figure()
                loss = h_phase.history['loss']
                plt.plot(loss)
                plt.savefig("./figure_res/loss_phase.png")
                with open(record_file, 'a+') as f:
                        f.write("@ train %s.\n" % (linenum))
                        f.write("@ train fault phases.\n")
                        f.write("@ snr = %s.\n" % (snr))
                        f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
                        f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
                        f.write("@ Now Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_val_acc] * 100, m_val_acc + 1))
                        f.write("@ Last Training Accuracy: %.2f %% .\n" % (acc[-1] * 100))
                        f.write("@ Last Testing Accuracy: %.2f %% \n." % (val_acc[-1] * 100))
            K.clear_session()
 
        ############ Location test #######################
        if faultloc == 1:
            epochs = 6000
            model_loc = build_model_location1(layers)
            input_label_train = label_type_train[:,-1]
            input_label_test = label_type_test[:,-1]
            global_start_time = time.time()
            h_loc = model_loc.fit(
                    input_data_train,
                    input_label_train,
                    batch_size=int(kinds*0.3),
                    validation_data=(input_data_test, input_label_test),
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
            if record:
                plt.figure()
                plt.plot(mae)
                plt.savefig("./figure_res/mae_loc.png")
    
                plt.figure()
                plt.plot(val_mae)
                plt.savefig("./figure_res/m_val_mae_loc.png")
    
                plt.figure()
                loss = h_loc.history['loss']
                plt.plot(loss)
                plt.savefig("./figure_res/loss_loc.png")
                with open(record_file, 'a+') as f:
                    f.write("@ train %s.\n" % (linenum))
                    f.write("@ train fault locations.\n")
                    f.write("@ snr = %s.\n" % (snr))
                    f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_mae] * 100, m_mae + 1))
                    f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_mae[m_val_mae] * 100, m_val_mae + 1))
                    f.write("@ Now Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_val_mae] * 100, m_val_mae + 1))
                    f.write("@ Last Training Accuracy: %.2f %% .\n" % (mae[-1] * 100))
                    f.write("@ Last Testing Accuracy: %.2f %% .\n" % (val_mae[-1] * 100))
            K.clear_session()
