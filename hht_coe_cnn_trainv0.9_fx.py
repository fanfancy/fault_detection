#log
#change the position of hht coe 0224, work with htt_coe_cnn_modelv09.py
import time
import numpy as np
from numpy import random as nr
import os
from tqdm import tqdm
import sys
import scipy.stats

import keras
from keras.models import Model
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras import backend as K
# from keras.callbacks import Reduc

from readdata import load_data
from httprocess2 import *
from matplotlib import pyplot as plt 
   

####read file ############
#linenum_all = ['313']
#linenum_all = ['49']
#linenum_all = ['75']
#linenum_all = ['49', '269', '313', '316', '75', '72', '69', '66']
linenum_all = ['49', '269',  '316', '75', '72', '69', '66','81', '319']
#linenum_all = ['313']

#linenum_all = ['313', '316', '75', '72', '69', '66']
# linenum_all = ['100002']
writefile = 'train_data'
# suffix = '_320point.txt'
# suffix = '_400point_large_resistance.txt'
# suffix = '_400point_4load_change.txt'
# suffix = '_1600point_randtime300.txt'
suffix = '_1600point_randtime300.txt'

### whhether save the result or not
record = 1
record_file = "log0304_netmode4.txt"

#dataformat parameters
length = 1600                               #length of the seuqence
diff_conditions = 1                         #load change nums
loc = 9                                     #loaction types
fault_type = 11                             #AG,BG,..,ABCG fault types
resistance = 4                              # resistance types [0.01, 0.1, 1, 10, 100 ...]
n_imfs = 2                                  # nimf nums

#########TODO#######################
repeat_times = 300
net_mode = 4
normal = diff_conditions
fault_kinds = fault_type
kinds = (fault_kinds + normal)*repeat_times # 12000 cycle
faulttype = 1
faultphase = 1
faultloc = 1

#training parameters
epochs = 2000

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
    from htt_coe_cnn_modelv09 import *
    layers = [length,2*n_imfs,3,n_imfs]
elif net_mode == 4:
    from htt_coe_cnn_modelv09 import *
    layers = [length,3,1,n_imfs]
elif net_mode == 5:
    from cnn_kernel_test_model import *
    layers = [length,2*n_imfs,3,n_imfs]
elif net_mode == 6:                        
    from cnn_model_2channel import *
    layers = [length,3,2]
elif net_mode == 7:                        
    from temp_cnn_model_2branch import *
    layers = [length,3,1]

#######################################################


for linenum in linenum_all:
    print(linenum," trainging test")
    curdir = os.path.abspath(os.curdir)
    filename = os.path.join(curdir,'./data/'+linenum+writefile+suffix)


    data, label, name = load_data(filename,length,kinds,fault_type)
    print ("data shape1",data.shape) #(19200000, 3)
    print ("label.shape",label.shape)
    label = label[0::length]
    print ("label.shape",label.shape)
    data = np.reshape(data,[-1,length,3])
    print ("data shape2",data.shape) #(12000, 1600, 3)

    #################### whether using HHT and its coe or not ###############
    if net_mode == 1:
        ### data for CNN only
        pass
    # else:
    res = multicore(data, n_imfs+1, kinds) #data with hht TODO check 6=2+4
    res = np.array(res) #(12000, 1600, 4, 3)    
    print ("res shape",res.shape)
    data2 = np.zeros((kinds, length, n_imfs * 2, 3))
    print ("data2 shape",data2.shape) #(12000, 1600, 2, 3)
    for i in range(kinds):
        data2[i, :, :, :] = res[i, :, 2:, :]

        # if net_mode == 3 or net_mode ==4:
    start = int(length * 0.25)
    end = -start
    HHT_coefficients = np.zeros((kinds,6,n_imfs,3))
    #HHT_coefficients = np.zeros((kinds,3,n_imfs,6))
    print ("HHT_coefficients shape",HHT_coefficients.shape) #(12000, 3, 2, 6)

    for kind in range(kinds):
        for i in range(n_imfs): #imf 0 1
            for j in range(3): #phase
                HHT_coefficients[kind,0,i,j] = np.max(data2[kind,start:end,2*i+1,j])
                HHT_coefficients[kind,1,i,j] = np.min(data2[kind,start:end,2*i+1,j])
                HHT_coefficients[kind,2,i,j] = np.mean(data2[kind,start:end,2*i+1,j])
                HHT_coefficients[kind,3,i,j] = np.std(data2[kind,start:end,2*i+1,j])
                HHT_coefficients[kind,4,i,j] = scipy.stats.skew(data2[kind,start:end,2*i+1,j])
                HHT_coefficients[kind,5,i,j] = np.sum(np.square(data2[kind,start:end,2*i+1,j]))


    HHT_coefficients_ori = np.reshape(HHT_coefficients,[kinds,-1])
    print ("HHT_coefficients_ori shape",HHT_coefficients_ori.shape) #(12000, 36)

    ################### adjust the format to input directly into CNN##############
    data3 = data[:,:,:,np.newaxis]

    
    #############shuffle the data #######################################
    index = [i for i in range(data.shape[0])]
    np.random.shuffle(index)
    data2_type = data2[index,:,:,:]
    data3_type = data3[index,:,:,:]
    print ("data2_type",data2_type.shape)
    HHT_coefficients_type = HHT_coefficients_ori[index,:]
    label = label[index,:]
    print ("label",label.shape)

    print ("data3_type",data3_type.shape)
    data2_type_reshape = data2_type[:,:,0,:].reshape(kinds,length,3,1)      #ifreq
    data2_type_reshape_2 = data2_type[:,:,1,:].reshape(kinds,length,3,1)    #amp
    print ("data2_type_reshape",data2_type_reshape.shape)

    data_2channel = np.concatenate((data2_type_reshape,data3_type),axis=3) #TODO
    # data_2channel = np.concatenate((data3_type,data2_type_reshape),axis=3) 
    print (data_2channel.shape)

    ################ choose the network structure ##############
    if net_mode==1:
        input_data = data3_type #data2_type_reshape #
    elif net_mode==2:
        input_data = data2_type
    elif net_mode==3:
        input_data = {'main_input':data2_type,'hht_co_input':HHT_coefficients_type}
    elif net_mode==4:
        input_data = {'main_input':data3_type,'hht_co_input':HHT_coefficients_type}
    elif net_mode==5:
        input_data = {'main_input':data2_type,'hht_co_input':HHT_coefficients_type}
    elif net_mode==6:
        input_data = {'main_input':data_2channel}
    elif net_mode==7:
        input_data = {'main_input':data3_type,'hht_ifreq_input':data2_type_reshape}
    ##################Type test ################################1
    if faulttype == 1:
        epochs = 2000
        input_label = label[:,0:6]
        global_start_time = time.time()
        model_type = build_model_type1(layers, fault_type)
        h_type = model_type.fit(
                    input_data,
                    input_label,
                    batch_size=int(kinds*0.1), 
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


        if record:
            #plt.figure()
            #plt.plot(acc)
            #plt.savefig("./figure_res/acc_type.png")

            #plt.figure()
            #plt.plot(val_acc)
            #plt.savefig("./figure_res/val_acc_type.png")

            #plt.figure()
            loss = h_type.history['loss']
            #plt.plot(loss)
            #plt.savefig("./figure_res/loss_type.png")


            with open(record_file,'a+') as f:
                f.write("@ train %s.\n" % (linenum))
                f.write("@ train fault types.\n")
                f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
                f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
                f.write("@ Now Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_val_acc] * 100, m_val_acc + 1))
                f.write("@ Last Training Accuracy: %.2f %% .\n" % (acc[-1] * 100))
                f.write("@ Last Testing Accuracy: %.2f %% .\n" % (val_acc[-1] * 100))

        K.clear_session()

############### Phase test ########################2
    if faultphase == 1:
        epochs = 2000
        model_phase = build_model_phase1(layers)
        input_label = label[:,6:10]
        global_start_time = time.time()
        h_phase = model_phase.fit(
                input_data,
                input_label,
                batch_size=int(kinds*0.1),
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
        if record:
            #plt.figure()
            #plt.plot(acc)
            #plt.savefig("./figure_res/acc_phase.png")

            #plt.figure()
            #plt.plot(val_acc)
            #plt.savefig("./figure_res/val_acc_phase.png")

            #plt.figure()
            loss = h_phase.history['loss']
            #plt.plot(loss)
            #plt.savefig("./figure_res/loss_phase.png")

            with open(record_file, 'a+') as f:
                f.write("@ train %s.\n" % (linenum))
                f.write("@ train fault phases.\n")
                f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
                f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
                f.write("@ Now Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_val_acc] * 100, m_val_acc + 1))
                f.write("@ Last Training Accuracy: %.2f %% .\n" % (acc[-1] * 100))
                f.write("@ Last Testing Accuracy: %.2f %% .\n" % (val_acc[-1] * 100))

        K.clear_session()

    ############ Location test #######################3
    if faultloc == 1:
        epochs = 3000
        model_loc = build_model_location1(layers)
        input_label = label[: ,-1]
        global_start_time = time.time()
        h_loc = model_loc.fit(
                input_data,
                input_label,
                batch_size=int(kinds*0.1), 
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
        if record:
            #plt.figure()
            #plt.plot(mae)
            #plt.savefig("./figure_res/mae_loc.png")

            #plt.figure()
            #plt.plot(val_mae)
            #plt.savefig("./figure_res/m_val_mae_loc.png")

            #plt.figure()
            loss = h_loc.history['loss']
            #plt.plot(loss)
            #plt.savefig("./figure_res/loss_loc.png")

            with open(record_file, 'a+') as f:
                f.write("@ train %s.\n" % (linenum))
                f.write("@ train fault locations.\n")
                f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_mae] * 100, m_mae + 1))
                f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_mae[m_val_mae] * 100, m_val_mae + 1))
                f.write("@ Now Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_val_mae] * 100, m_val_mae + 1))
                f.write("@ Last Training Accuracy: %.2f %% .\n" % (mae[-1] * 100))
                f.write("@ Last Testing Accuracy: %.2f %% .\n" % (val_mae[-1] * 100))
    
        K.clear_session()
