from keras.models import load_model
from keras import Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, Input
from keras.optimizers import Adam

if __name__ == '__main__':
    
    # Load existing model
    model = load_model('models/neo_character_model_v4.h5')
    model.pop()
    model.pop()
    # Add more neurons to existing dense layers
    model.layers[-4].units = 128  # Increase first dense layer to 128 neurons
    model.layers[-2].units = 64  # Increase second dense layer to 64 neurons

    # Optional: Add additional layer
    # model.layers.insert(-1, Dense(64, activation='relu'))
    # model.layers.insert(-1, Dropout(0.3))

    # Recompile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Save expanded model
    model.save('models/neo_character_model_v7.h5')

    print("Model expanded and saved successfully!")