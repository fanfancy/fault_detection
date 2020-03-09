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



def build_model_type(input,fault_type):
    
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    x = Conv2D(32, (10, 1), activation='relu',padding='same')(main_input)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(16, (10, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    hht_co_input = Input(shape=([input[3]*18]),name='hht_co_input')
    x = keras.layers.core.Reshape([-1])(x)
    x = keras.layers.concatenate([x,hht_co_input],axis=1)
    x = Dense(128,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    x = Dense(64,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    output = Dense(fault_type-5,activation='softmax',name = 'output')(x)
    
    model = Model(inputs=[main_input,hht_co_input],outputs=output)
    start = time.time()
    model.compile(loss=losses.categorical_crossentropy, optimizer='Adam', metrics=['acc','mse', 'mae'])

    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_type1(input,fault_type):
    
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(main_input)
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x) # 6*3*32
    hht_co_input = Input(shape=([input[3]*18]),name='hht_co_input')
    x = keras.layers.core.Reshape([-1])(x)

    x = keras.layers.concatenate([x,hht_co_input],axis=1)



    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer = l2(0.0003))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer = l2(0.0003))(x)
    output = Dense(fault_type-5,activation='softmax', name = 'output', kernel_regularizer = l2(0.0003))(x)
    
    model = Model(inputs=[main_input,hht_co_input],outputs=output)
    start = time.time()
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=losses.categorical_crossentropy, optimizer='Adam',metrics=['acc','mse', 'mae']) #'Adam'
    model.summary()

    print("> Compilation Time : ", time.time() - start)
    return model


def build_model_phase(input):
    
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    x = Conv2D(32, (10, 1), activation='relu',padding='same')(main_input)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(16, (10, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    hht_co_input = Input(shape=([input[3]*18]),name='hht_co_input')
    x = keras.layers.core.Reshape([-1])(x)
    x = keras.layers.concatenate([x,hht_co_input],axis=1)
    x = Dense(128,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    x = Dense(64,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    output = Dense(4,activation='sigmoid',name = 'output')(x)
    
    model = Model(inputs=[main_input,hht_co_input],outputs=output)
    start = time.time()
    model.compile(loss=losses.binary_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_phase1(input):
    
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(main_input)
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    hht_co_input = Input(shape=([input[3]*18]),name='hht_co_input')
    x = keras.layers.core.Reshape([-1])(x)
    x = keras.layers.concatenate([x,hht_co_input],axis=1)

    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0003))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0003))(x)
    output = Dense(4,activation='sigmoid', name = 'output', kernel_regularizer=l2(0.0003))(x)
    
    model = Model(inputs=[main_input,hht_co_input],outputs=output)
    start = time.time()
    model.compile(loss=losses.binary_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])
    model.summary()

    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_location(input):
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    x = Conv2D(32, (10, 1), activation='relu',padding='same')(main_input)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(16, (10, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    hht_co_input = Input(shape=([input[3]*18]),name='hht_co_input')
    x = keras.layers.core.Reshape([-1])(x)
    x = keras.layers.concatenate([x,hht_co_input],axis=1)
    x = Dense(128,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    x = Dense(64,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    output = Dense(1,activation='sigmoid',name = 'output')(x)
    
    model = Model(inputs=[main_input,hht_co_input],outputs=output)
    start = time.time()
    model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_location1(input):
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(main_input)
    x = Conv2D(4, (16, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)
    x = Conv2D(8, (16, 1), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)
    x = Conv2D(16, (16, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)
    x = Conv2D(32, (16, 1), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(4, 1))(x)

    hht_co_input = Input(shape=([input[3]*18]),name='hht_co_input')
    x = keras.layers.core.Reshape([-1])(x)
    x = keras.layers.concatenate([x,hht_co_input],axis=1)


    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0003))(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.0003))(x)
    output = Dense(1,activation='sigmoid', name = 'output', kernel_regularizer=l2(0.0003))(x)
    
    model = Model(inputs=[main_input,hht_co_input],outputs=output)
    start = time.time()
    model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])
    model.summary()
    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_location2(input):
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    x = Conv1D(4, 6, activation='relu', padding='same')(main_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(8, 16, activation='relu', padding='same')(x)
    x = Conv1D(8, 16, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Conv1D(16, 16, activation='relu',padding='same')(x)
    x = Conv1D(16, 16, activation='relu',padding='same')(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Conv1D(16, 16, activation='relu',padding='same')(x)
    x = Conv1D(16, 16, activatiion='relu',padding='same')(x)
    x = MaxPooling1D(pool_size=4)(x)

    hht_co_input = Input(shape=([input[3]*18]),name='hht_co_input')
    x = keras.layers.core.Reshape([-1])(x)
    x = keras.layers.concatenate([x,hht_co_input],axis=1)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1,activation='sigmoid', name = 'output')(x)
    
    model = Model(inputs=[main_input,hht_co_input],outputs=output)
    start = time.time()
    model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])
    model.summary()
    print("> Compilation Time : ", time.time() - start)
    return model
