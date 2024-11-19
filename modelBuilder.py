from keras.models import Sequential, load_model as keras_load_model
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D

# TODO : actually derive this value. i'm gonna just assume it's true.
unique_characters_list = ['Q', 'W', '-', 'C', 'D', 'E', 'Z', 'T', 'L', 'R', 'J', 'G', 'S', ' ', 'Y', 'X', 'B', 'V', "'", 'I', 'N', 'P', 'O', 'A', 'M', 'K', 'F', 'U', 'H']


def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    model = keras_load_model(filename)
    return model

# def create_model(input_shape, num_classes):
#     model = Sequential()
#     model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
#     model.add(BatchNormalization())  # Batch normalization after Conv2D
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes, activation='softmax'))
#     return model
    # model = Sequential()

    # model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=input_shape))
    # model.add(MaxPooling2D())

    # model.add(Conv2D(32, (3,3), 1, activation='relu'))
    # model.add(MaxPooling2D())

    # model.add(Conv2D(16, (3,3), 1, activation='relu'))
    # model.add(MaxPooling2D())

    # model.add(Flatten())
    # model.add(Dense(256, activation='relu'))
    # model.add(Dense(len(unique_characters_list), activation='softmax'))
    # return model
    
def create_model(input_shape, num_classes):
    model = Sequential([
        # First Convolutional Block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    return model