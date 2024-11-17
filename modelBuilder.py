import pickle
from tensorflow.keras.models import load_model

def save_model(model, filename):
    """Save the model to a pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model_from_pickle(filename):
    """Load the model from a pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def create_model(input_shape, num_classes):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Reshape((-1, 64)))  # Reshape for LSTM (depends on your image size)
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(128, return_sequences=True))
    
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
