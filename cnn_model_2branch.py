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

def build_model_type1(input,fault_type):
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input') 	#1600*3*1
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(main_input)           #1600*3*4 
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(x)                    #1600*3*4
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #400*3*4
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)                    #400*3*8
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)                    #400*3*8
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #100*3*8
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)                    #100*3*16
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)                    #100*3*16
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #25*3*16
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)                    #25*3*32
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)                    #25*3*32
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #6*3*32
    
    hht_input = Input(shape=([input[3], input[4], input[5]]), name='hht_input')
    f = Conv2D(4, (8, 1), activation='relu', padding='same')(hht_input)	 	        #800*3*2
    f = MaxPooling2D(pool_size=(4, 1))(f) 					                        #200*3*4
    f = Conv2D(8, (8, 1), activation='relu', padding='same')(f)			            #200*3*8
    f = MaxPooling2D(pool_size=(4, 1))(f)					                        #50*3*8
    f = Conv2D(16, (8, 1), activation='relu',padding='same')(f) 			        #50*3*16
    f = MaxPooling2D(pool_size=(4, 1))(f)					                        #12*3*16
    f = Conv2D(32, (8, 1), activation='relu',padding='same')(f) 			        #12*3*32
    f = MaxPooling2D(pool_size=(4, 1))(f)       				                    #3*3*32

    y = keras.layers.concatenate([x,f],axis=1)
    y = keras.layers.core.Reshape([-1])(y)

    y = Dropout(0.5)(y) 
    y = Dense(64, activation='relu', kernel_regularizer = l2(0.0003))(y)
    y = Dropout(0.5)(y)
    y = Dense(64, activation='relu', kernel_regularizer = l2(0.0003))(y)
    output = Dense(fault_type-5,activation='softmax', name = 'output', kernel_regularizer = l2(0.0003))(y)
    model = Model(inputs=[main_input, hht_input], outputs=output)
    start = time.time()
    model.compile(loss=losses.categorical_crossentropy, optimizer='Adam', metrics=['acc','mse', 'mae'])
    model.summary()

    print("> Compilation Time : ", time.time() - start)
    return model


def build_model_phase1(input):
    
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input') 	#1600*3*1
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(main_input)           #1600*3*4 
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(x)                    #1600*3*4
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #400*3*4
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)                    #400*3*8
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)                    #400*3*8
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #100*3*8
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)                    #100*3*16
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)                    #100*3*16
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #25*3*16
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)                    #25*3*32
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)                    #25*3*32
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #6*3*32
    
    hht_input = Input(shape=([input[3], input[4], input[5]]), name='hht_input')
    f = Conv2D(4, (8, 1), activation='relu', padding='same')(hht_input)	 	        #800*3*4
    f = MaxPooling2D(pool_size=(4, 1))(f) 					                        #200*3*4
    f = Conv2D(8, (8, 1), activation='relu', padding='same')(f)			            #200*3*8
    f = MaxPooling2D(pool_size=(4, 1))(f)					                        #50*3*8
    f = Conv2D(16, (8, 1), activation='relu',padding='same')(f) 			         #50*3*16
    f = MaxPooling2D(pool_size=(4, 1))(f)					                        #12*3*16
    f = Conv2D(32, (8, 1), activation='relu',padding='same')(f) 			        #12*3*32
    f = MaxPooling2D(pool_size=(4, 1))(f)       				                    #3*3*32

    # x = keras.layers.core.Reshape([-1])(x)
    # f = keras.layers.core.Reshape([-1])(f)
    y = keras.layers.concatenate([x,f],axis=1)
    y = keras.layers.core.Reshape([-1])(x)

    y = Dropout(0.5)(y) 
    y = Dense(64, activation='relu', kernel_regularizer = l2(0.0003))(y)
    y = Dropout(0.5)(y)
    y = Dense(64, activation='relu', kernel_regularizer = l2(0.0003))(y)
    output = Dense(3,activation='sigmoid', name = 'output', kernel_regularizer = l2(0.0003))(y) #changed 0308，3output

    model = Model(inputs=[main_input, hht_input], outputs=output)
    start = time.time()
    model.compile(loss=losses.binary_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])
    model.summary()

    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_location1(input):
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input') 	#1600*3*1
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(main_input)           #1600*3*4 
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(x)                    #1600*3*4
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #400*3*4
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)                    #400*3*8
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)                    #400*3*8
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #100*3*8
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)                    #100*3*16
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)                    #100*3*16
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #25*3*16
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)                    #25*3*32
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)                    #25*3*32
    x = MaxPooling2D(pool_size=(4, 1))(x)                                           #6*3*32
    
    hht_input = Input(shape=([input[3], input[4], input[5]]), name='hht_input')
    f = Conv2D(4, (8, 1), activation='relu', padding='same')(hht_input)	 	        #800*3*2
    f = MaxPooling2D(pool_size=(4, 1))(f) 					                        #200*3*4
    f = Conv2D(8, (8, 1), activation='relu', padding='same')(f)			            #200*3*8
    f = MaxPooling2D(pool_size=(4, 1))(f)					                        #50*3*8
    f = Conv2D(16, (8, 1), activation='relu',padding='same')(f) 			        #50*3*16
    f = MaxPooling2D(pool_size=(4, 1))(f)					                        #12*3*16
    f = Conv2D(32, (8, 1), activation='relu',padding='same')(f) 			        #12*3*32
    f = MaxPooling2D(pool_size=(4, 1))(f)       				                    #3*3*32

    y = keras.layers.concatenate([x,f],axis=1)
    y = keras.layers.core.Reshape([-1])(y)

    y = Dropout(0.5)(y) 
    y = Dense(64, activation='relu', kernel_regularizer = l2(0.0003))(y)
    y = Dropout(0.5)(y)
    y = Dense(64, activation='relu', kernel_regularizer = l2(0.0003))(y)
    output = Dense(1,activation='sigmoid', name = 'output', kernel_regularizer = l2(0.0003))(y)
   
    model = Model(inputs=[main_input, hht_input], outputs=output)
    start = time.time()
    model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])
    model.summary()
    print("> Compilation Time : ", time.time() - start)
    return model
