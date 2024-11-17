from tensorflow.keras import layers, models

def create_model(input_shape, num_classes):
    model = models.Sequential()
    
    # CNN layers for feature extraction
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the CNN output and add LSTM for sequential learning
    model.add(layers.Reshape((-1, 64)))  # Reshape for LSTM (depends on your image size)
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(128, return_sequences=True))
    
    # Dense output layer for predictions
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
