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
   

####read file ############
#linenum_all = ['313']
linenum_all = ['49']
# linenum_all = ['49', '269', '313', '316', '75', '72', '69', '66']
# linenum_all = ['100002']
writefile = 'train_data'
# suffix = '_320point.txt'
suffix = '_400point_large_resistance.txt'



### whhether save the result or not
record = 1

#dataformat parameters
length = 400                            #length of the seuqence
diff_conditions = 8                       #load change nums
loc = 9                               #loaction types
fault_type = 10                          #AG,BG,..,ABCG fault types
resistance = 6                          # resistance types [0.01, 0.1, 1, 10, 100 ...]
n_imfs = 2                             # nimf nums
repeat_times = resistance * loc
normal = diff_conditions * repeat_times
fault_kinds = loc*fault_type*resistance*diff_conditions
kinds = fault_kinds + normal


#training parameters
epochs = 5000
#####work mode################################################
'''
net_mode = 1  using current data directly
net_mode = 2  using HHT processed data without HHT coe
net_mode = 3  using HHT processed data with HHT coe
net_mode = 4  using current data directly with HHT coe
net_mode = 5  cnn kernel test
'''
net_mode = 4
if net_mode == 1:
    from cnn_model import *
    layers = [length,3,1,n_imfs]
elif net_mode == 2 :
    from cnn_model import *
    layers = [length,2*n_imfs,3,n_imfs]
elif net_mode == 3 :
    from htt_coe_cnn_model import *
    layers = [length,2*n_imfs,3,n_imfs]
elif net_mode == 4:
    from htt_coe_cnn_model import *
    layers = [length,3,1,n_imfs]
elif net_mode == 5:
    from cnn_kernel_test_model import *
    layers = [length,2*n_imfs,3,n_imfs]

#######################################################


for linenum in linenum_all:
    print(linenum," trainging test")
    curdir = os.path.abspath(os.curdir)
    filename = os.path.join(curdir,'./data/'+linenum+writefile+suffix)


    data, label, name = load_data(filename,length,kinds)
    label = label[0::length]
    data = np.reshape(data,[-1,length,3])

    #################### whether using HHT and its coe or not ###############
    if net_mode == 1:
        ### data for CNN only
        pass
    # else:
    res = multicore(data, n_imfs, kinds)
    data2 = np.zeros((kinds, length, n_imfs * 2, 3))
    for i in range(kinds):
        data2[i, :, :, :] = res[i]

        # if net_mode == 3 or net_mode ==4:
    start = int(length * 0.25)
    end = -start
    HHT_coefficients = np.zeros((kinds,6,2*n_imfs,3))
    import scipy.stats

    for kind in range(kinds):
        for i in range(2*n_imfs):
            for j in range(3):
                HHT_coefficients[kind,0,i,j] = np.max(          data2[kind,start:end,i,j])
                HHT_coefficients[kind,1,i,j] = np.min(          data2[kind,start:end,i,j])
                HHT_coefficients[kind,2,i,j] = np.mean(         data2[kind,start:end,i,j])
                HHT_coefficients[kind,3,i,j] = np.std(          data2[kind,start:end,i,j])
                HHT_coefficients[kind,4,i,j] = scipy.stats.skew(   data2[kind,start:end,i,j])
                HHT_coefficients[kind,5,i,j] = np.sum(np.square(   data2[kind,start:end,i,j]))


    HHT_coefficients_ori = np.reshape(HHT_coefficients,[kinds,-1])

    ################### adjust the format to input directly into CNN##############
    data3 = data[:,:,:,np.newaxis]

    
    #############shuffle the data #######################################
    index = [i for i in range(data.shape[0])]
    np.random.shuffle(index)
    data2_type = data2[index,:,:,:]
    data3_type = data3[index,:,:,:]
    HHT_coefficients_type = HHT_coefficients_ori[index,:]
    label_type = label[index,:]
    print (label_type.shape)
    ################ choose the network structure ##############
    if net_mode==1:
        input_data = data3_type
    elif net_mode==2:
        input_data = data2_type
    elif net_mode==3:
        input_data = {'main_input':data2_type,'hht_co_input':HHT_coefficients_type}
    elif net_mode==4:
        input_data = {'main_input':data3_type,'hht_co_input':HHT_coefficients_type}
    elif net_mode==5:
        input_data = {'main_input':data2_type,'hht_co_input':HHT_coefficients_type}

    ##################Type test ################################1
    input_label = label_type[:,0:5]
    global_start_time = time.time()
    model_type = build_model_type(layers)
    h_type = model_type.fit(
                input_data,
                input_label,
                batch_size=int(kinds*0.75),
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
        with open("log_htt2.txt",'a+') as f:
            f.write("@ train %s.\n" % (linenum))
            f.write("@ train fault types.\n")
            f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
            f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
            f.write("@ Last Training Accuracy: %.2f %% .\n" % (acc[-1] * 100))
            f.write("@ Last Testing Accuracy: %.2f %% .\n" % (val_acc[-1] * 100))

    K.clear_session()

################ Phase test ########################2
#    
#    #############shuffle the data #######################################
#    index_phase = [i for i in range(fault_kinds)]
#    np.random.shuffle(index_phase)
#    data2_phase = data2[index_phase,:,:,:]
#    data3_phase = data3[index_phase,:,:,:]
#    HHT_coefficients_phase = HHT_coefficients_ori[index_phase,:]
#    label_phase = label[index_phase,:]
#    print (label_phase.shape)
#    
#    
#    if net_mode==1:
#        input_data = data3_phase
#    elif net_mode==2:
#        input_data = data2_phase
#    elif net_mode==3:
#        input_data = {'main_input':data2_phase,'hht_co_input':HHT_coefficients_phase}
#    elif net_mode==4:
#        input_data = {'main_input':data3_phase,'hht_co_input':HHT_coefficients_phase}
#    elif net_mode==5:
#        input_data = {'main_input':data2_phase,'hht_co_input':HHT_coefficients_phase}
#    
#    model_phase = build_model_phase(layers)
#    input_label = label_phase[:,4:8]
#    global_start_time = time.time()
#    h_phase = model_phase.fit(
#            input_data,
#            input_label,
#            batch_size=int(kinds*0.75),
#            validation_split=0.25,
#            verbose=0,
#            shuffle=True,
#            epochs=epochs)
#    
#    print('Phase Training duration (s) : ', time.time() - global_start_time)
#    acc = h_phase.history['acc']
#    val_acc = h_phase.history['val_acc']
#    print("@ Last Training Accuracy: %.2f %% ." % (acc[-1] * 100))
#    print("@ Last Testing Accuracy: %.2f %% ." % (val_acc[-1] * 100))
#    m_acc = np.argmax(acc)
#    m_val_acc = np.argmax(val_acc)
#    print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
#    print("@ Best Testing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
#    if record:
#        with open("log_htt2.txt", 'a+') as f:
#            f.write("@ train %s.\n" % (linenum))
#            f.write("@ train fault phases.\n")
#            f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
#            f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
#    K.clear_session()
#
#    ############ Location test #######################3
#    model_loc = build_model_location(layers)
#    input_label = label_phase[: ,-1]
#    global_start_time = time.time()
#    h_loc = model_loc.fit(
#            input_data,
#            input_label,
#            batch_size=int(kinds*0.75),
#            validation_split=0.25,
#            verbose=0,
#            shuffle=True,
#            epochs=epochs)
#    print('Location Training duration (s) : ', time.time() - global_start_time)
#    mae = h_loc.history['mean_absolute_error']
#    val_mae = h_loc.history['val_mean_absolute_error']
#    m_mae = np.argmin(mae)
#    m_val_mae = np.argmin(val_mae)
#    print("@ Best Training mae: %.2f %% achieved at EP #%d." % (mae[m_mae] * 100, m_mae + 1))
#    print("@ Best Testing mae: %.2f %% achieved at EP #%d." % (val_mae[m_val_mae] * 100, m_val_mae + 1))
#    print("@ Last Training Accuracy: %.2f %% ." % (mae[-1] * 100))
#    print("@ Last Testing Accuracy: %.2f %% ." % (val_mae[-1] * 100))
#    if record:
#        with open("log_htt2.txt", 'a+') as f:
#            f.write("@ train %s.\n" % (linenum))
#            f.write("@ train fault locations.\n")
#            f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_mae] * 100, m_mae + 1))
#            f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_mae[m_val_mae] * 100, m_val_mae + 1))
#    K.clear_session()
