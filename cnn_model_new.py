from keras import losses
from keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D
import time
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dense,Dropout,GRU
from keras.models import Model
from keras import regularizers
from keras.regularizers import l2
#from keras.layers import BatchNormalization
import keras



def build_model_type(input,fault_type):
    model = Sequential()
    model.add(Conv2D(32, (10, 1), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(16, (10, 1), activation='relu', padding='same'))
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

def build_model_type1(input,fault_type):
    model = Sequential()
    model.add(Conv2D(4, (16, 1), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
    model.add(Conv2D(4, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(8, (16, 1), activation='relu', padding='same'))
    model.add(Conv2D(8, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(16, (16, 1), activation='relu', padding='same'))
    model.add(Conv2D(16, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(32, (16, 1), activation='relu', padding='same'))
    model.add(Conv2D(32, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
        
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0003)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0003)))
    model.add(Dense(fault_type-5, activation='softmax', kernel_regularizer=l2(0.0003)))

    start = time.time()
    model.compile(loss=losses.categorical_crossentropy, \
                  optimizer="Adam", metrics=['acc', 'mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_type2(input,fault_type):
    model = Sequential()
    model.add(Conv1D(8, 10, activation='relu', padding='same', input_shape=(input[0], input[1])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(16, 10, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(16, 10, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, 10, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))        
    model.add(Conv1D(32, 10, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(fault_type-5, activation='softmax'))

    start = time.time()
    model.compile(loss=losses.categorical_crossentropy, \
                  optimizer="Adam", metrics=['acc', 'mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_phase(input):
    model = Sequential()
    model.add(Conv2D(32, (10, 1), activation='relu', padding='same',\
                     input_shape=(input[0], input[1], input[2])))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(16, (10, 1), activation='relu', padding='same'))
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

def build_model_phase1(input):
    model = Sequential()
    model.add(Conv2D(4, (16, 1), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
    model.add(Conv2D(4, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(8, (16, 1), activation='relu', padding='same'))
    model.add(Conv2D(8, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(16, (16, 1), activation='relu', padding='same'))
    model.add(Conv2D(16, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(32, (16, 1), activation='relu', padding='same'))
    model.add(Conv2D(32, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
        
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0003)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0003)))
    model.add(Dense(4, activation='sigmoid', kernel_regularizer=l2(0.0003)))
    
    start = time.time()
    model.compile(loss=losses.binary_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_location(input):
    model = Sequential()
    model.add(Conv2D(32, (10, 1), activation='relu', padding='same',\
                     input_shape=(input[0], input[1], input[2])))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(Conv2D(16, (10, 1), activation='relu', padding='same'))
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

def build_model_location1(input):
    model = Sequential()
    model.add(Conv2D(4, (16, 1), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
    model.add(Conv2D(4, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(8, (16, 1), activation='relu', padding='same'))
    model.add(Conv2D(8, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(16, (16, 1), activation='relu', padding='same'))
    model.add(Conv2D(16, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
    model.add(Conv2D(32, (16, 1), activation='relu', padding='same'))
    model.add(Conv2D(32, (16, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 1)))
        
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0003)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.0003)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0003)))

    start = time.time()
    model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])
    print("> Compilation Time : ", time.time() - start)
    return model
