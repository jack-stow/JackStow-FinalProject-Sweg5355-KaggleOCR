import pickle
from keras.models import Sequential, load_model as keras_load_model
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization

def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    model = keras_load_model(filename)
    return model

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())  # Batch normalization after Conv2D
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model