import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense

def save_model(model, filename):
    """Save the model to a pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model_from_pickle(filename):
    """Load the model from a pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def create_model(input_shape, num_classes):
    model = Sequential()  # Fixed models to Sequential
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Reshape((-1, 64)))  # Reshape for LSTM (depends on your image size)
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
