#log
#change the position of hht coe 0224, work with htt_coe_cnn_modelv0.9_fx.py

from keras import losses
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, Conv1D, MaxPooling1D
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense,Dropout,GRU
from keras.models import Model
from keras.regularizers import l2
import keras
import sys
import numpy as np

def build_model_type1(input,fault_type):
    data1 = Input(shape=([input[0], input[1], input[2]]), name='inputa')  
    
    x = Conv2D(2, (3, 3), activation='relu', padding='same')(data1)	 	    #100*100*1
    x = MaxPooling2D(pool_size=(2, 2))(x) 					                        #50 *50 *3
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)			            #50 *50 *6
    x = MaxPooling2D(pool_size=(2, 2))(x)					                        #25 *25 *6
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)			            #25 *25 *8
    x = MaxPooling2D(pool_size=(2, 2))(x)					                        #12 *12 *8
    x = Conv2D(16, (3, 3), activation='relu',padding='same')(x) 			        #12 *12 *16
    x = MaxPooling2D(pool_size=(2, 2))(x)					                        #6  *6  *16
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x) 			        #6  *6  *32
    x = MaxPooling2D(pool_size=(2, 2))(x) 			                                #3  *3  *32
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(x) 			        #6  *6  *32
    x = MaxPooling2D(pool_size=(2, 2))(x) 			                                #1  *1  *64
    x = keras.layers.core.Reshape([-1])(x)
    
    data2 = Input(shape=([input[0], input[1], input[2]]), name='inputb')  
    y = Conv2D(2, (3, 3), activation='relu', padding='same')(data2)	 	            #100*100*1
    y = MaxPooling2D(pool_size=(2, 2))(y) 					                        #50 *50 *3
    y = Conv2D(4, (3, 3), activation='relu', padding='same')(y)			            #50 *50 *6
    y = MaxPooling2D(pool_size=(2, 2))(y)					                        #25 *25 *6
    y = Conv2D(8, (3, 3), activation='relu', padding='same')(y)			            #25 *25 *8
    y = MaxPooling2D(pool_size=(2, 2))(y)					                        #12 *12 *8
    y = Conv2D(16, (3, 3), activation='relu',padding='same')(y) 			        #12 *12 *16
    y = MaxPooling2D(pool_size=(2, 2))(y)					                        #6  *6  *16
    y = Conv2D(32, (3, 3), activation='relu',padding='same')(y) 			        #6  *6  *32
    y = MaxPooling2D(pool_size=(2, 2))(y) 			                                #3  *3  *32
    y = Conv2D(64, (3, 3), activation='relu',padding='same')(y) 			        #6  *6  *32
    y = MaxPooling2D(pool_size=(2, 2))(y) 			                                #1  *1  *64
    y = keras.layers.core.Reshape([-1])(y)

    data3 = Input(shape=([input[0], input[1], input[2]]), name='inputc')  
    z = Conv2D(2, (3, 3), activation='relu', padding='same')(data3)	 	    #100*100*1
    z = MaxPooling2D(pool_size=(2, 2))(z) 					                        #50 *50 *3
    z = Conv2D(4, (3, 3), activation='relu', padding='same')(z)			            #50 *50 *6
    z = MaxPooling2D(pool_size=(2, 2))(z)					                        #25 *25 *6
    z = Conv2D(8, (3, 3), activation='relu', padding='same')(z)			            #25 *25 *8
    z = MaxPooling2D(pool_size=(2, 2))(z)					                        #12 *12 *8
    z = Conv2D(16, (3, 3), activation='relu',padding='same')(z) 			        #12 *12 *16
    z = MaxPooling2D(pool_size=(2, 2))(z)					                        #6  *6  *16
    z = Conv2D(32, (3, 3), activation='relu',padding='same',kernel_regularizer = l2(0.01))(z) 			        #6  *6  *32
    z = MaxPooling2D(pool_size=(2, 2))(z) 			                                #3  *3  *32
    z = Conv2D(64, (3, 3), activation='relu',padding='same',kernel_regularizer = l2(0.01))(z) 			        #6  *6  *32
    z = MaxPooling2D(pool_size=(2, 2))(z) 			                                #1  *1  *64
    z = keras.layers.core.Reshape([-1])(z)


    x = keras.layers.concatenate([x,y,z],axis=1)
    x = Dropout(0.3)(x)     
    x = Dense(64, activation='relu', kernel_regularizer = l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer = l2(0.01))(x)
    output = Dense(fault_type-5,activation='softmax', name = 'output', kernel_regularizer = l2(0.01))(x)
    model = Model(inputs=[data1,data2,data3],outputs=output)
    start = time.time()
    model.compile(loss=losses.categorical_crossentropy, optimizer='Adam', metrics=['acc','mse', 'mae'])
    model.summary()
    #sys.exit()

    print("> Compilation Time : ", time.time() - start)
    return model


def build_model_phase1(input):
    data1 = Input(shape=([input[0], input[1], input[2]]), name='inputa')  
    
    x = Conv2D(2, (3, 3), activation='relu', padding='same')(data1)	 	    #100*100*1
    x = MaxPooling2D(pool_size=(2, 2))(x) 					                        #50 *50 *3
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)			            #50 *50 *6
    x = MaxPooling2D(pool_size=(2, 2))(x)					                        #25 *25 *6
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)			            #25 *25 *8
    x = MaxPooling2D(pool_size=(2, 2))(x)					                        #12 *12 *8
    x = Conv2D(16, (3, 3), activation='relu',padding='same')(x) 			        #12 *12 *16
    x = MaxPooling2D(pool_size=(2, 2))(x)					                        #6  *6  *16
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x) 			        #6  *6  *32
    x = MaxPooling2D(pool_size=(2, 2))(x) 			                                #3  *3  *32
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(x) 			        #6  *6  *32
    x = MaxPooling2D(pool_size=(2, 2))(x) 			                                #1  *1  *64
    x = keras.layers.core.Reshape([-1])(x)
    
    data2 = Input(shape=([input[0], input[1], input[2]]), name='inputb')  
    y = Conv2D(2, (3, 3), activation='relu', padding='same')(data2)	 	            #100*100*1
    y = MaxPooling2D(pool_size=(2, 2))(y) 					                        #50 *50 *3
    y = Conv2D(4, (3, 3), activation='relu', padding='same')(y)			            #50 *50 *6
    y = MaxPooling2D(pool_size=(2, 2))(y)					                        #25 *25 *6
    y = Conv2D(8, (3, 3), activation='relu', padding='same')(y)			            #25 *25 *8
    y = MaxPooling2D(pool_size=(2, 2))(y)					                        #12 *12 *8
    y = Conv2D(16, (3, 3), activation='relu',padding='same')(y) 			        #12 *12 *16
    y = MaxPooling2D(pool_size=(2, 2))(y)					                        #6  *6  *16
    y = Conv2D(32, (3, 3), activation='relu',padding='same')(y) 			        #6  *6  *32
    y = MaxPooling2D(pool_size=(2, 2))(y) 			                                #3  *3  *32
    y = Conv2D(64, (3, 3), activation='relu',padding='same')(y) 			        #6  *6  *32
    y = MaxPooling2D(pool_size=(2, 2))(y) 			                                #1  *1  *64
    y = keras.layers.core.Reshape([-1])(y)

    data3 = Input(shape=([input[0], input[1], input[2]]), name='inputc')  
    z = Conv2D(2, (3, 3), activation='relu', padding='same')(data3)	 	    #100*100*1
    z = MaxPooling2D(pool_size=(2, 2))(z) 					                        #50 *50 *3
    z = Conv2D(4, (3, 3), activation='relu', padding='same')(z)			            #50 *50 *6
    z = MaxPooling2D(pool_size=(2, 2))(z)					                        #25 *25 *6
    z = Conv2D(8, (3, 3), activation='relu', padding='same')(z)			            #25 *25 *8
    z = MaxPooling2D(pool_size=(2, 2))(z)					                        #12 *12 *8
    z = Conv2D(16, (3, 3), activation='relu',padding='same')(z) 			        #12 *12 *16
    z = MaxPooling2D(pool_size=(2, 2))(z)					                        #6  *6  *16
    z = Conv2D(32, (3, 3), activation='relu',padding='same',kernel_regularizer = l2(0.01))(z) 			        #6  *6  *32
    z = MaxPooling2D(pool_size=(2, 2))(z) 			                                #3  *3  *32
    z = Conv2D(64, (3, 3), activation='relu',padding='same',kernel_regularizer = l2(0.01))(z) 			        #6  *6  *32
    z = MaxPooling2D(pool_size=(2, 2))(z) 			                                #1  *1  *64
    z = keras.layers.core.Reshape([-1])(z)


    x = keras.layers.concatenate([x,y,z],axis=1)
    x = Dropout(0.3)(x)     #TODO
    x = Dense(64, activation='relu', kernel_regularizer = l2(0.0003))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer = l2(0.0003))(x)
    output = Dense(3,activation='sigmoid', name = 'output', kernel_regularizer=l2(0.0003))(x)
    
    model = Model(inputs=[data1,data2,data3],outputs=output)
    start = time.time()
    model.compile(loss=losses.binary_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])
    model.summary()

    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_location1(input):
    data1 = Input(shape=([input[0], input[1], input[2]]), name='inputa')  
    
    x = Conv2D(2, (3, 3), activation='relu', padding='same')(data1)	 	    #100*100*1
    x = MaxPooling2D(pool_size=(2, 2))(x) 					                        #50 *50 *3
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)			            #50 *50 *6
    x = MaxPooling2D(pool_size=(2, 2))(x)					                        #25 *25 *6
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)			            #25 *25 *8
    x = MaxPooling2D(pool_size=(2, 2))(x)					                        #12 *12 *8
    x = Conv2D(16, (3, 3), activation='relu',padding='same')(x) 			        #12 *12 *16
    x = MaxPooling2D(pool_size=(2, 2))(x)					                        #6  *6  *16
    x = Conv2D(32, (3, 3), activation='relu',padding='same')(x) 			        #6  *6  *32
    x = MaxPooling2D(pool_size=(2, 2))(x) 			                                #3  *3  *32
    x = Conv2D(64, (3, 3), activation='relu',padding='same')(x) 			        #6  *6  *32
    x = MaxPooling2D(pool_size=(2, 2))(x) 			                                #1  *1  *64
    x = keras.layers.core.Reshape([-1])(x)
    
    data2 = Input(shape=([input[0], input[1], input[2]]), name='inputb')  
    y = Conv2D(2, (3, 3), activation='relu', padding='same')(data2)	 	            #100*100*1
    y = MaxPooling2D(pool_size=(2, 2))(y) 					                        #50 *50 *3
    y = Conv2D(4, (3, 3), activation='relu', padding='same')(y)			            #50 *50 *6
    y = MaxPooling2D(pool_size=(2, 2))(y)					                        #25 *25 *6
    y = Conv2D(8, (3, 3), activation='relu', padding='same')(y)			            #25 *25 *8
    y = MaxPooling2D(pool_size=(2, 2))(y)					                        #12 *12 *8
    y = Conv2D(16, (3, 3), activation='relu',padding='same')(y) 			        #12 *12 *16
    y = MaxPooling2D(pool_size=(2, 2))(y)					                        #6  *6  *16
    y = Conv2D(32, (3, 3), activation='relu',padding='same')(y) 			        #6  *6  *32
    y = MaxPooling2D(pool_size=(2, 2))(y) 			                                #3  *3  *32
    y = Conv2D(64, (3, 3), activation='relu',padding='same')(y) 			        #6  *6  *32
    y = MaxPooling2D(pool_size=(2, 2))(y) 			                                #1  *1  *64
    y = keras.layers.core.Reshape([-1])(y)

    data3 = Input(shape=([input[0], input[1], input[2]]), name='inputc')  
    z = Conv2D(2, (3, 3), activation='relu', padding='same')(data3)	 	    #100*100*1
    z = MaxPooling2D(pool_size=(2, 2))(z) 					                        #50 *50 *3
    z = Conv2D(4, (3, 3), activation='relu', padding='same')(z)			            #50 *50 *6
    z = MaxPooling2D(pool_size=(2, 2))(z)					                        #25 *25 *6
    z = Conv2D(8, (3, 3), activation='relu', padding='same')(z)			            #25 *25 *8
    z = MaxPooling2D(pool_size=(2, 2))(z)					                        #12 *12 *8
    z = Conv2D(16, (3, 3), activation='relu',padding='same')(z) 			        #12 *12 *16
    z = MaxPooling2D(pool_size=(2, 2))(z)					                        #6  *6  *16
    z = Conv2D(32, (3, 3), activation='relu',padding='same',kernel_regularizer = l2(0.01))(z) 			        #6  *6  *32
    z = MaxPooling2D(pool_size=(2, 2))(z) 			                                #3  *3  *32
    z = Conv2D(64, (3, 3), activation='relu',padding='same',kernel_regularizer = l2(0.01))(z) 			        #6  *6  *32
    z = MaxPooling2D(pool_size=(2, 2))(z) 			                                #1  *1  *64
    z = keras.layers.core.Reshape([-1])(z)


    x = keras.layers.concatenate([x,y,z],axis=1)
    x = Dropout(0.3)(x)     #TODO
    x = Dense(64, activation='relu', kernel_regularizer = l2(0.0003))(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer = l2(0.0003))(x)
    output = Dense(1,activation='sigmoid', name = 'output', kernel_regularizer=l2(0.0003))(x)
    
    model = Model(inputs=[data1,data2,data3],outputs=output)
    start = time.time()
    model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])
    model.summary()
    print("> Compilation Time : ", time.time() - start)
    return model
