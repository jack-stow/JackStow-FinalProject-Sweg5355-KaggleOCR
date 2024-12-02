from keras.models import load_model
from keras import Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.models import Sequential


# Function to create the expanded model
def create_expanded_model(input_shape, num_classes, learning_rate=0.0001):
    model = Sequential([
        # First Convolutional Block
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third Convolutional Block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense Layers
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        
        # Output layer with softmax for multi-class classification
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

import numpy as np

def transfer_weights_with_expansion(old_layer, new_layer):
    old_weights = old_layer.get_weights()
    new_weights = new_layer.get_weights()

    if not old_weights or not new_weights:
        return  # Skip layers without weights

    try:
        for i in range(len(new_weights)):
            # Expand weight shapes as necessary
            if old_weights[i].shape == new_weights[i].shape:
                new_weights[i] = old_weights[i]  # Direct copy if shapes match
            else:
                # Handle weight expansion
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_weights[i].shape, new_weights[i].shape))
                new_weights[i][slices] = old_weights[i][slices]  # Copy existing weights
                print(f"Expanded weights for layer {new_layer.name}")
        
        new_layer.set_weights(new_weights)
    except ValueError as e:
        print(f"Skipping layer {new_layer.name} due to shape mismatch: {e}")

if __name__ == '__main__':
    input_shape = (32, 32, 3)
    num_classes = 49

    # Load existing model
    model = load_model('models/neo_character_model_v4.h5')
    model2 = create_expanded_model(input_shape, num_classes, learning_rate=0.0001)
    
    # Transfer weights for matching layers with expansion
    for old_layer, new_layer in zip(model.layers, model2.layers):
        transfer_weights_with_expansion(old_layer, new_layer)
    
    # Save expanded model
    model2.save('models/neo_character_model_v5.h5')
    print("Model expanded and saved successfully!")
