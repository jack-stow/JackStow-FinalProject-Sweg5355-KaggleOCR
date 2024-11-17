from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from dataLoader import load_data
from modelBuilder import create_model
from labelEncoder import encode_labels
from mlDataset import path

print("================")
print(path)
print("================")
# Load and preprocess the data
train_images, train_labels = load_data('train_v2/train/', path=path)
encoded_labels, label_encoder = encode_labels(train_labels)

# One-hot encode the labels (for categorical output)
train_labels_one_hot = to_categorical(encoded_labels, num_classes=len(label_encoder.classes_))

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels_one_hot, test_size=0.2, random_state=42)

# Define model
model = create_model(X_train.shape[1:], len(label_encoder.classes_))

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)
