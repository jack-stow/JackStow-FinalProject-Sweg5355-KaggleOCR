from keras.models import load_model
from keras import Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, Input
from keras.optimizers import Adam
from modelBuilder import create_model

input_shape = (32, 32, 3)  # Update to include channels (assuming RGB input)
num_classes = 49

def create_original_model():
    model = create_model(input_shape, num_classes, learning_rate=0.0001)
    return model

def revert_model():
    try:
        # Load the current, modified model
        model = load_model('models/neo_character_model_v4.h5')

        # Create the original model
        original_model = create_original_model()

        # Load the weights from the modified model
        original_model.set_weights(model.get_weights())

        # Save the reverted model
        original_model.save('models/neo_character_model_v4_reverted.h5')
        print("Model reverted successfully and saved as 'neo_character_model_v4_reverted.h5'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    revert_model()
