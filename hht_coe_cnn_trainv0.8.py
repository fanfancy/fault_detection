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

from httprocess2 import *
from keras.models import load_model
from sklearn.model_selection import train_test_split

####read file ############
#linenum_all = ['313']
#linenum_all = ['49']
#linenum_all = ['75']
#linenum_all = ['269', '313', '316', '75', '72', '69', '66','81', '319'] #49
linenum_all = ['66','81', '319']
#linenum_all = ['313']

#linenum_all = ['313', '316', '75', '72', '69', '66']
# linenum_all = ['100002']
writefile = 'train_data'
# suffix = '_320point.txt'
# suffix = '_400point_large_resistance.txt'
# suffix = '_400point_4load_change.txt'
suffix = '_1600point_randtime300.txt'
# suffix = '_1600point_randtime1000.txt'

### whhether save the result or not
record = 1

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


#training parameters
epochs = 2000
faulttype = 0
faultphase = 1
faultloc = 0
#####work mode################################################
'''
net_mode = 1  using current data directly
net_mode = 2  using HHT processed data without HHT coe
net_mode = 3  using HHT processed data with HHT coe
net_mode = 4  using current data directly with HHT coe
net_mode = 5  cnn kernel test
'''
net_mode = 1

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

#######################################################
for linenum in linenum_all:
    print(linenum," trainging test")
    curdir = os.path.abspath(os.curdir)
    filename = os.path.join(curdir,'./data/'+linenum+writefile+suffix)


    data, label, name = load_data(filename,length,kinds,fault_type)
    label = label[0::length]
    data = np.reshape(data,[-1,length,3])
    print ("data.shape",data.shape)

    #################### whether using HHT and its coe or not ###############
    if net_mode == 1:
        ### data for CNN only
        pass
    # else:
    res = multicore(data, 3, kinds)
    res = np.array(res)
    print (res.shape)
    data2 = np.zeros((kinds, length, n_imfs * 2, 3))
    for i in range(kinds):
        data2[i, :, :, :] = res[i, :, 2:, :]

        # if net_mode == 3 or net_mode ==4:
    start = int(length * 0.25)
    end = -start
    HHT_coefficients = np.zeros((kinds,6,n_imfs,3))
    import scipy.stats

    for kind in range(kinds):
        for i in range(n_imfs):
            for j in range(3):
                HHT_coefficients[kind,0,i,j] = np.max(          data2[kind,start:end,2*i+1,j])
                HHT_coefficients[kind,1,i,j] = np.min(          data2[kind,start:end,2*i+1,j])
                HHT_coefficients[kind,2,i,j] = np.mean(         data2[kind,start:end,2*i+1,j])
                HHT_coefficients[kind,3,i,j] = np.std(          data2[kind,start:end,2*i+1,j])
                HHT_coefficients[kind,4,i,j] = scipy.stats.skew(   data2[kind,start:end,2*i+1,j])
                HHT_coefficients[kind,5,i,j] = np.sum(np.square(   data2[kind,start:end,2*i+1,j]))


    HHT_coefficients_ori = np.reshape(HHT_coefficients,[kinds,-1])

    ################### adjust the format to input directly into CNN##############
    data3 = data[:,:,:,np.newaxis]

    
    #############shuffle the data #######################################
    index = [i for i in range(data.shape[0])]
    np.random.shuffle(index)
    data2_type = data2[index,:,:,:]
    data3_type = data3[index,:,:,:]
    HHT_coefficients_type = HHT_coefficients_ori[index,:]
    label = label[index,:]
    print ("label.shape",label.shape)
    name = np.array(name)
    print ("name len",len(name))
    name = name[index]
    print ("name.shape",name.shape)

    # split dataset
    data2_type_train = data2_type[:int(kinds*0.75)]
    data2_type_test  = data2_type[int(kinds*0.75):]
    data3_type_train = data3_type[:int(kinds*0.75)]
    data3_type_test  = data3_type[int(kinds*0.75):]
    HHT_coe_train = HHT_coefficients_type[:int(kinds*0.75)]
    HHT_coe_test  = HHT_coefficients_type[int(kinds*0.75):]
    label_train = label[:int(kinds*0.75)]
    label_test  = label[int(kinds*0.75):]

    print ("data2_type_train.shape",data2_type_train.shape)
    print ("data2_type_test.shape",data2_type_test.shape)
    print ("data3_type_train.shape",data3_type_train.shape)
    print ("data3_type_test.shape",data3_type_test.shape)
    print ("HHT_coe_train.shape",HHT_coe_train.shape)
    print ("HHT_coe_test.shape",HHT_coe_test.shape)
    print ("label_train.shape",label_train.shape)
    print ("label_test.shape",label_test.shape)

    

    ################ choose the network structure ##############
    if net_mode==1:
        input_data = data3_type
        data_train = data3_type_train
        data_test = data3_type_test
    elif net_mode==2:
        input_data = data2_type
    elif net_mode==3:
        input_data = {'main_input':data2_type,'hht_co_input':HHT_coefficients_type}
    elif net_mode==4:
        input_data = {'main_input':data3_type,'hht_co_input':HHT_coefficients_type}
        data_train = {'main_input':data3_type_train,'hht_co_input':HHT_coe_train}
        data_test = {'main_input':data3_type_test,'hht_co_input':HHT_coe_test}
    elif net_mode==5:
        input_data = {'main_input':data2_type,'hht_co_input':HHT_coefficients_type}

    
   
    ##################Type test ################################1
    if faulttype == 1:
        epochs = 2000
        input_label = label[:,0:6]
        label_train_current_net = label_train[:,0:6]
        label_test_current_net = label_test[:,0:6]
        global_start_time = time.time()
        model_type = build_model_type1(layers, fault_type)
        h_type = model_type.fit(
                    data_train,
                    label_train_current_net,
                    batch_size=int(kinds*0.1), 
                    validation_data=(data_test, label_test_current_net),#validation_split=0.25,
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
            with open("0306_rand300_log_netmode1_phase.txt",'a+') as f:
                f.write("@ train %s.\n" % (linenum))
                f.write("@ train fault types.\n")
                f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
                f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
                f.write("@ Now Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_val_acc] * 100, m_val_acc + 1))
                f.write("@ Last Training Accuracy: %.2f %% .\n" % (acc[-1] * 100))
                f.write("@ Last Testing Accuracy: %.2f %% .\n" % (val_acc[-1] * 100))
        ########################################################
        ##  model_type.save("0305_netmode1_type.h5")
        ##  testmodel = load_model("0305_netmode1_type.h5")
        ##  predict = model_type.predict(input_data)
        ##  a = np.argmax(predict, axis=1)
        ##  b = np.argmax(input_label,axis=1)
        ##  print ("predict.shape",predict.shape)
        ##  false = []
        ##  for ii in range(predict.shape[0]):
        ##      if not(a[ii]==b[ii]):
        ##          false.append(ii)
        ##  print("false",false)
        ##  print("false len",len(false))
        ##  print ("input_label.shape",input_label.shape)
        ##  with open("0306_rand300_log_netmode1_phase_error_type.txt",'w') as f:
        ##          for i in range (len(false)):
        ##              print(name[false[i]],"\t\t",input_label[false[i]],"\t\t",predict[false[i]],file=f)

        K.clear_session()

############### Phase test ########################2
    if faultphase == 1:
        epochs = 1000
        model_phase = build_model_phase1(layers)
        input_label = label[:,6:9] #[:,6:10] #changed to ABC 3 phases 0306
        label_train_current_net = label_train[:,6:9]
        label_test_current_net = label_test[:,6:9]
        label_test_current_net = label_test_current_net.astype(np.int16) #for comparing in phase test
        label_train_current_net = label_train_current_net.astype(np.int16)

        global_start_time = time.time()
        h_phase = model_phase.fit(
                data_train,
                label_train_current_net,
                batch_size=int(kinds*0.1),
                validation_data=(data_test, label_test_current_net),#validation_split=0.25,
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
            with open("0306_rand300_log_netmode1_phase.txt", 'a+') as f:
                f.write("@ train %s.\n" % (linenum))
                f.write("@ train fault phases.\n")
                f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_acc] * 100, m_acc + 1))
                f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_acc[m_val_acc] * 100, m_val_acc + 1))
                f.write("@ Now Training Accuracy: %.2f %% achieved at EP #%d.\n" % (acc[m_val_acc] * 100, m_val_acc + 1))
                f.write("@ Last Training Accuracy: %.2f %% .\n" % (acc[-1] * 100))
                f.write("@ Last Testing Accuracy: %.2f %% .\n" % (val_acc[-1] * 100))
        
        ############train phase accuracy###############################
        model_phase.save("0305_netmode1_phase.h5")

        testmodel = load_model("0305_netmode1_phase.h5")
        predict = model_phase.predict(data_train) #900*3
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0
        predict=predict.astype(np.int16)
        print ("predict.shape",predict.shape)
        false = []
        for ii in range(predict.shape[0]):
            #print ("type(predict[ii][0])",type(predict[ii][0]))
            #print ("type(label_test_current_net[ii][0])",type(label_train_current_net[ii][0]))
            if not(all(predict[ii]==label_train_current_net[ii])):
                false.append(ii)

        print("len(false)",len(false))
        print("false",false)
        with open("0306_rand300_log_netmode1_phase.txt", 'a+') as f:
            print ("train_accuracy",1-len(false)/2700,file=f)
            for i in range (len(false)):
                print(label_train_current_net[false[i]],"\t\t",predict[false[i]],file=f)

        ######valid phase accuracy#############
        testmodel = load_model("0305_netmode1_phase.h5")
        predict = model_phase.predict(data_test) #900*3
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0
        predict=predict.astype(np.int16)
        print ("predict.shape",predict.shape)
        false = []
        for ii in range(predict.shape[0]):
            #print ("type(predict[ii][0])",type(predict[ii][0]))
            #print ("type(label_test_current_net[ii][0])",type(label_test_current_net[ii][0]))
            if not(all(predict[ii]==label_test_current_net[ii])):
                false.append(ii)

        print("len(false)",len(false))
        print("false",false)
        print ("label_test_current_net.shape",label_test_current_net.shape)
        with open("0306_rand300_log_netmode1_phase.txt", 'a+') as f:
            print ("valid_accuracy",1-len(false)/900,file=f)
            for i in range (len(false)):
                print(label_test_current_net[false[i]],"\t\t",predict[false[i]],file=f)

        K.clear_session()

    ############ Location test #######################3
    if faultloc == 1:
        epochs = 3000
        model_loc = build_model_location1(layers)
        input_label = label[: ,-1]
        label_train_current_net = label_train[: ,-1]
        label_test_current_net =label_test[: ,-1]
        global_start_time = time.time()
        h_loc = model_loc.fit(
                data_train,
                label_train_current_net,
                batch_size=int(kinds*0.1),
                validation_data=(data_test, label_test_current_net),#validation_split=0.25,
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
            with open("0306_rand300_log_netmode1_phase.txt", 'a+') as f:
                f.write("@ train %s.\n" % (linenum))
                f.write("@ train fault locations.\n")
                f.write("@ Best Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_mae] * 100, m_mae + 1))
                f.write("@ Best Testing Accuracy: %.2f %% achieved at EP #%d.\n" % (val_mae[m_val_mae] * 100, m_val_mae + 1))
                f.write("@ Now Training Accuracy: %.2f %% achieved at EP #%d.\n" % (mae[m_val_mae] * 100, m_val_mae + 1))
                f.write("@ Last Training Accuracy: %.2f %% .\n" % (mae[-1] * 100))
                f.write("@ Last Testing Accuracy: %.2f %% .\n" % (val_mae[-1] * 100))

        K.clear_session()
