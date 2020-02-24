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

# def build_model_type(input):
#     model = Sequential()
#     # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
#     # this applies 32 convolution filters of size 3x3 each.

#     model.add(Conv2D(32, (10, 2), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
#     # model.add(Conv2D(8, (5, 1), activation='relu',padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 1)))
#     # model.add((keras.layers.normalization.BatchNormalization(axis=1)))
#     # model.add(Conv2D(32, (10, 2), activation='relu',padding='same'))
#     # model.add(Conv2D(32, (5, 1), activation='relu', padding='same'))

#     # model.add(MaxPooling2D(pool_size=(2, 1)))
#     # model.add(Dropout(0.25))

#     # model.add(Conv2D(16, (10, 2), activation='relu', padding='same'))
#     model.add(Conv2D(16, (10, 2), activation='relu', padding='same'))
#     # model.add(Conv2D(64, (10, 2), activation='relu', padding='same'))
#     # model.add(Conv2D(16, (5, 1), activation='relu', padding='same'))
#     # model.add(Conv2D(8, (1, 1), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 1)))
#     # model.add((keras.layers.normalization.BatchNormalization(axis=1)))
#     # model.add(Conv2D(8, (1, 1), activation='relu', padding='same'))
#     # model.add(MaxPooling2D(pool_size=(2, 1)))
#     # model.add(Dropout(0.25))
#     # model.add(Conv2D(128, (11, 3), activation='relu', padding='same'))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#     #     model.add(Reshape((80, 64)))
#     #     model.add(GRU(128))

#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))

#     model.add(Dense(4, activation='softmax'))

#     start = time.time()
#     model.compile(loss=losses.categorical_crossentropy, optimizer="Adam", metrics=['acc', 'mse', 'mae'])
#     print("> Compilation Time : ", time.time() - start)
#     return model

def build_model_type(input):
#     model = Sequential()
#     # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
#     # this applies 32 convolution filters of size 3x3 each.

#     model.add(Conv2D(32, (10, 2), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
#     model.add(MaxPooling2D(pool_size=(2, 1)))
#     model.add(Conv2D(16, (10, 2), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 1)))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(4, activation='softmax'))

#     start = time.time()
#     model.compile(loss=losses.categorical_crossentropy, optimizer="Adam", metrics=['acc', 'mse', 'mae'])
#     print("> Compilation Time : ", time.time() - start)
    # return model
#     main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    
#     x = Conv2D(32, (3, 2), activation='relu',padding='same')(main_input)
#     x = Conv2D(32, (3, 2), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
#     x = MaxPooling2D(pool_size=(2, 1))(x)
#     x = Conv2D(32, (3, 2), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
#     x = Conv2D(32, (3, 2), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
#     x = MaxPooling2D(pool_size=(2, 1))(x)
#     x = Conv2D(16, (3, 2), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
#     x = Conv2D(16, (3, 2), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
#     x = MaxPooling2D(pool_size=(2, 1))(x)
#     x = Conv2D(16, (3, 2), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
#     x = Conv2D(16, (3, 2), activation='relu',padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
#     x = MaxPooling2D(pool_size=(2, 1))(x)
    
#     hht_co_input = Input(shape=([72]),name='hht_co_input')
#     x = keras.layers.core.Reshape([-1])(x)
    
#     x = keras.layers.concatenate([x,hht_co_input],axis=1)
    
#     x = Dense(128,activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)
#     x = Dropout(0.3)(x)
#     x = Dense(64,activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)
#     x = Dropout(0.3)(x)
    
#     output = Dense(4,activation='softmax',name = 'output',kernel_regularizer=regularizers.l2(0.01))(x)
    
#     model = Model(inputs=[main_input,hht_co_input],outputs=output)
   
#     start = time.time()
    
#     model.compile(loss=losses.categorical_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])
#     model.compile(loss=losses.binary_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])
#     model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])
    
    main_input = Input(shape=([input[0], input[1], input[2]]), name='main_input')
    
    x = Conv2D(32, (10, 2), activation='relu',padding='same')(main_input)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(16, (10, 2), activation='relu',padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    
    hht_co_input = Input(shape=([72]),name='hht_co_input')
    x = keras.layers.core.Reshape([-1])(x)
    
    x = keras.layers.concatenate([x,hht_co_input],axis=1)
    
    x = Dense(128,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    x = Dense(64,activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    
    output = Dense(5,activation='softmax',name = 'output')(x)
    
    model = Model(inputs=[main_input,hht_co_input],outputs=output)
   
    start = time.time()
    
    model.compile(loss=losses.categorical_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])

    print("> Compilation Time : ", time.time() - start)
    return model
def build_model_phase(input):
    model = Sequential()
    model.add(Conv2D(32, (10, 2), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
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
    model.add(Conv2D(32, (10, 2), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
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

# def build_model_phase(input):
#     model = Sequential()
#     # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
#     # this applies 32 convolution filters of size 3x3 each.

#     model.add(Conv2D(32, (11, 3), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
#     #     model.add(Conv2D(32, (5, 1), activation='relu',padding='same'))
#     #     model.add(Conv2D(32, (5, 5), activation='relu',padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 1)))
#     # model.add(Dropout(0.25))

#     model.add(Conv2D(16, (11, 3), activation='relu', padding='same'))
#     # model.add(Conv2D(64, (5, 1), activation='relu', padding='same'))
#     #     model.add(Conv2D(64, (2, 2), activation='relu',padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 1)))
#     # model.add(Dropout(0.25))

#     #     model.add(Reshape((80, 64)))
#     #     model.add(GRU(128))

#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))

#     model.add(Dense(4, activation='sigmoid'))



#     start = time.time()
#     model.compile(loss=losses.binary_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])
#     print("> Compilation Time : ", time.time() - start)
#     return model


# def build_model_location(input):
#     model = Sequential()
#     # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
#     # this applies 32 convolution filters of size 3x3 each.

#     model.add(Conv2D(32, (10, 1), activation='relu', padding='same', input_shape=(input[0], input[1], input[2])))
#     model.add(MaxPooling2D(pool_size=(2, 1)))
#     model.add(Dropout(0.25))

#     model.add(Conv2D(64, (5, 1), activation='relu', padding='same'))
#     model.add(Conv2D(64, (5, 1), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 1)))
#     model.add(Dropout(0.25))

#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.3))
#     model.add(Dense(1, activation='sigmoid'))

#     start = time.time()
#     model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])
#     print("> Compilation Time : ", time.time() - start)
#     return model
