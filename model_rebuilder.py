
from keras.models import load_model
from keras import Model
from keras.layers import Dense
from keras.optimizers import Adam
    

# I use(d) this file as a lazy way to rebuild my model.
# I started with only 10 classes, so i had to adjust it later to accept 62 classes.
if __name__ == '__main__':
    # Load the pre-trained model
    model = load_model('models/best_character_model_v0.h5')  # type: ignore

    # Access the second-to-last layer's output
    model_input = model.input
    x = model.layers[-2].output

    # Add a new output layer for 62 classes with a unique name
    new_output = Dense(62, activation='softmax', name='new_dense_output')(x)

    # Create a new model
    new_model = Model(inputs=model_input, outputs=new_output)

    # Compile the new model
    new_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',  # Changed to categorical_crossentropy
        metrics=['accuracy']
    )

    # Save the new model
    new_model.save('models/best_character_model_v1.h5')