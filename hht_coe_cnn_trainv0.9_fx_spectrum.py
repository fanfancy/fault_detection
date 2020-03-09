#log
#for spectrum  100*100 
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

from readdata_spectrum import load_data
from matplotlib import pyplot as plt 
from cnn_model_spectrum import * 

####read file ############
#linenum_all = ['49', '269', '313', '316', '75', '72', '69', '66']
#linenum_all = ['81', '319']
linenum_all = ['313']
writefile = 'train_data'
# suffix = '_320point.txt'
# suffix = '_400point_large_resistance.txt'
# suffix = '_400point_4load_change.txt'
# suffix = '_1600point_randtime300.txt'
suffix = '_1600point_randtime300_spectrum.txt'
suffix_label = '_1600point_randtime300_label.txt'

### whhether save the result or not
record = 1
record_file = "0308_spectrum_phase.txt"

#dataformat parameters
length = 1600                               #length of the seuqence
diff_conditions = 1                         #load change nums
loc = 9                                     #loaction types
fault_type = 11                             #AG,BG,..,ABCG fault types
resistance = 4                              # resistance types [0.01, 0.1, 1, 10, 100 ...]
n_imfs = 2                                  # nimf nums

#########TODO#######################
repeat_times = 300
normal = diff_conditions
fault_kinds = fault_type
kinds = (fault_kinds + normal)*repeat_times # 12000 cycle
faulttype = 0
faultphase = 0
faultloc = 1

#training parameters
epochs = 2000


#####work mode################################################
layers = [100,100,1]  # 100 * 100 *3

for linenum in linenum_all:
    print(linenum," trainging test")
    curdir = os.path.abspath(os.curdir)
    filename = os.path.join(curdir,'./data/spectrum/'+linenum+writefile+suffix)
    labelfile = os.path.join(curdir,'./data/spectrum/'+linenum+writefile+suffix_label)


    data, label, name = load_data(filename,labelfile,kinds,fault_type)
    print ("data shape1",data.shape)  #(3600, 100, 100,3)

    #data = data[:,:,:,np.newaxis]
    #for xx in range (label.shape[0]):
    #    print (xx,'\t',label[xx])
    #sys.exit()

    #############shuffle the data #######################################
    index = [i for i in range(data.shape[0])] #3600
    np.random.shuffle(index)
    data_each_kind = data[index,:,:,:]
    print ("data_each_kind",data_each_kind.shape)
    label = label[index,:]
    print ("label",label.shape)
    
    data_a = data_each_kind[:,:,:,0]
    data_b = data_each_kind[:,:,:,1]
    data_c = data_each_kind[:,:,:,2]
    data_a = data_a[:,:,:,np.newaxis]
    data_b = data_b[:,:,:,np.newaxis]
    data_c = data_c[:,:,:,np.newaxis]
    
    label_train  = label [:int(kinds*0.75)]
    label_test   = label [int(kinds*0.75):]
    data_a_train = data_a[:int(kinds*0.75)]
    data_a_test  = data_a[int(kinds*0.75):]
    data_b_train = data_b[:int(kinds*0.75)]
    data_b_test  = data_b[int(kinds*0.75):]
    data_c_train = data_c[:int(kinds*0.75)]
    data_c_test  = data_c[int(kinds*0.75):]

    ################ choose the network structure ##############
    input_data = {'inputa':data_a,'inputb':data_b,'inputc':data_c}
    data_train = {'inputa':data_a_train,'inputb':data_b_train,'inputc':data_c_train}
    data_test = {'inputa':data_a_test,'inputb':data_b_test,'inputc':data_c_test}
    
    ##################Type test ################################1
    if faulttype == 1:
        epochs = 3000
        input_label = label[:,0:6]
        label_test_current_net  = label_test [:,0:6]
        label_train_current_net = label_train[:,0:6]

        global_start_time = time.time()
        model_type = build_model_type1(layers, fault_type)
        h_type = model_type.fit(
                    data_train,
                    label_train_current_net,
                    batch_size=int(kinds*0.1), 
                    validation_data=(data_test, label_test_current_net),
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
        input_label = label[:,6:9]
        label_train_current_net = label_train[:,6:9]
        label_test_current_net = label_test[:,6:9]
        label_test_current_net = label_test_current_net.astype(np.int16) #for comparing in phase test
        label_train_current_net = label_train_current_net.astype(np.int16)

        global_start_time = time.time()
        h_phase = model_phase.fit(
                data_train,
                label_train_current_net,
                batch_size=int(kinds*0.4),
                validation_data=(data_test, label_test_current_net),
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
            ############train phase accuracy###############################
            model_phase.save("0308_spectrum_phase.h5")

            testmodel = load_model("0308_spectrum_phase.h5")
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
            with open(record_file, 'a+') as f:
                print ("train_accuracy",1-len(false)/2700,file=f)
                for i in range (len(false)):
                    print(label_train_current_net[false[i]],"\t\t",predict[false[i]],file=f)

            ######valid phase accuracy#############
            testmodel = load_model("0308_spectrum_phase.h5")
            predict = model_phase.predict(data_test) #900*3
            predict[predict >= 0.5] = 1
            predict[predict < 0.5] = 0
            predict=predict.astype(np.int16)
            print ("predict.shape",predict.shape)
            false = []
            for ii in range(predict.shape[0]):
                if not(all(predict[ii]==label_test_current_net[ii])):
                    false.append(ii)

            print("len(false)",len(false))
            print("false",false)
            print ("label_test_current_net.shape",label_test_current_net.shape)
            with open(record_file, 'a+') as f:
                print ("valid_accuracy",1-len(false)/900,file=f)
                for i in range (len(false)):
                    print(label_test_current_net[false[i]],"\t\t",predict[false[i]],file=f)

        K.clear_session()

    ############ Location test #######################3
    if faultloc == 1:
        epochs = 2000
        model_loc = build_model_location1(layers)
        input_label = label[: ,-1]
        label_test_current_net  = label_test [:,-1]
        label_train_current_net = label_train[:,-1]
        
        global_start_time = time.time()
        h_loc = model_loc.fit(
                data_train,
                label_train_current_net,
                batch_size=int(kinds*0.4), 
                validation_data=(data_test, label_test_current_net),
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
