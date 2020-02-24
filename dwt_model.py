import time
import keras
from keras.models import Sequential
from keras.layers import Input, Dense,Dropout,GRU
from keras.models import Model
from keras import losses


def build_model_type(layers):

    main_input = Input(shape=(layers[1],layers[0]), name='main_input')

    lstm1 = GRU(layers[3],return_sequences=True)(main_input)
    lstm2 = GRU(layers[2],return_sequences=True)(lstm1)
    lstm3 = GRU(layers[2],return_sequences=True)(lstm2)
    lstm4 = GRU(layers[2],return_sequences=False)(lstm3)

    dwt_input = Input(shape=(layers[-1],), name='dwt_input')

    x = keras.layers.concatenate([lstm4, dwt_input], axis=1)
    x = Dense(layers[2], activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    x = Dense(layers[3], activation='sigmoid')(x)
    x = Dropout(0.3)(x)

    output = Dense(layers[4], activation='softmax', name='output')(x)

    model = Model(inputs=[main_input, dwt_input], outputs=output)

    start = time.time()

    model.compile(loss=losses.categorical_crossentropy, optimizer="Adam", metrics=['acc', 'mse', 'mae'])

    print("> Compilation Time : ", time.time() - start)
    return model

def build_model_phase(layers):

    main_input = Input(shape=(layers[1],layers[0]), name='main_input')

    lstm1 = GRU(layers[3],return_sequences=True)(main_input)
    lstm2 = GRU(layers[2],return_sequences=True)(lstm1)
    lstm3 = GRU(layers[2],return_sequences=True)(lstm2)
    lstm4 = GRU(layers[2],return_sequences=False)(lstm3)

    dwt_input = Input(shape=(layers[-1],),name='dwt_input')

    x = keras.layers.concatenate([lstm4,dwt_input],axis=1)

    x = Dense(layers[2],activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    x = Dense(layers[3],activation='sigmoid')(x)
    x = Dropout(0.3)(x)


    output = Dense(layers[4],activation='sigmoid',name = 'output')(x)

    model = Model(inputs=[main_input,dwt_input],outputs=output)

    start = time.time()

    model.compile(loss=losses.binary_crossentropy, optimizer="Adam",metrics=['acc','mse', 'mae'])

    print("> Compilation Time : ", time.time() - start)
    return model


def build_model_location(layers):

    main_input = Input(shape=(layers[1],layers[0]), name='main_input')

    lstm1 = GRU(layers[3],return_sequences=True)(main_input)
    lstm2 = GRU(layers[2],return_sequences=True)(lstm1)
    lstm3 = GRU(layers[2],return_sequences=True)(lstm2)
    lstm4 = GRU(layers[2],return_sequences=False)(lstm3)

    dwt_input = Input(shape=(layers[-1],),name='dwt_input')

    x = keras.layers.concatenate([lstm4,dwt_input],axis=1)

    x = Dense(layers[2],activation='sigmoid')(x)
    x = Dropout(0.3)(x)
    x = Dense(layers[3],activation='sigmoid')(x)
    x = Dropout(0.3)(x)

    output = Dense(layers[4],activation='sigmoid',name = 'output')(x)

    model = Model(inputs=[main_input,dwt_input],outputs=output)

    start = time.time()

    model.compile(loss=losses.mean_squared_error, optimizer="Adam",metrics=['mse', 'mae'])

    print("> Compilation Time : ", time.time() - start)
    return model