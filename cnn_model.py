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



def build_model_type(input,fault_type):
    model = Sequential()
    model.add(Conv2D(32, (10, 2), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
    print ("input_shape",input_shape)
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(16, (10, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(fault_type-5, activation='softmax'))

    start = time.time()
    model.compile(loss=losses.categorical_crossentropy, \
                  optimizer="Adam", metrics=['acc', 'mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_phase(input):
    model = Sequential()
    model.add(Conv2D(32, (10, 2), activation='relu', padding='same',\
                     input_shape=(input[0], input[1], input[2])))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(16, (10, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(4, activation='sigmoid'))



    start = time.time()
    model.compile(loss=losses.binary_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model


def build_model_location(input):
    model = Sequential()
    model.add(Conv2D(32, (10, 2), activation='relu', padding='same',\
                     input_shape=(input[0], input[1], input[2])))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(16, (10, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    start = time.time()
    model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model
