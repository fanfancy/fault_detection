from keras import losses
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense,Dropout,GRU
from keras.models import Model
from keras import regularizers
import keras



def build_model_type(input):
    
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    x = Conv2D(32, (10, 2), activation='relu',padding='same')(main_input)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(16, (10, 2), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    dwt_co_input = Input(shape=([input[3]]),name='dwt_co_input')
    x = keras.layers.core.Reshape([-1])(x)
    x = keras.layers.concatenate([x, dwt_co_input],axis=1)
    x = Dense(128,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    x = Dense(64,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    output = Dense(5,activation='softmax',name = 'output')(x)
    
    model = Model(inputs=[main_input,dwt_co_input],outputs=output)
    start = time.time()
    model.compile(loss=losses.categorical_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])

    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_phase(input):
    
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    x = Conv2D(32, (10, 2), activation='relu',padding='same')(main_input)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(16, (10, 2), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    dwt_co_input = Input(shape=([input[3]]),name='dwt_co_input')
    x = keras.layers.core.Reshape([-1])(x)
    x = keras.layers.concatenate([x,dwt_co_input],axis=1)
    x = Dense(128,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    x = Dense(64,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    output = Dense(4,activation='sigmoid',name = 'output')(x)
    
    model = Model(inputs=[main_input,dwt_co_input],outputs=output)
    start = time.time()
    model.compile(loss=losses.binary_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model



def build_model_location(input):
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    x = Conv2D(32, (10, 2), activation='relu',padding='same')(main_input)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(16, (10, 2), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    dwt_co_input = Input(shape=([input[3]]),name='dwt_co_input')
    x = keras.layers.core.Reshape([-1])(x)
    x = keras.layers.concatenate([x,dwt_co_input],axis=1)
    x = Dense(128,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    x = Dense(64,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    output = Dense(1,activation='sigmoid',name = 'output')(x)
    
    model = Model(inputs=[main_input,dwt_co_input],outputs=output)
    start = time.time()
    model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model
