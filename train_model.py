import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint


IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 3 # Reduced for quick demonstration
MODEL_PATH = 'data/model/land_use_classifier.h5'
NUM_CLASSES = 4
CLASS_NAMES = ['Agriculture', 'Forest', 'Urban', 'Water']
DATA_MOCK_DIR = 'data/mock_training_data'

def create_mock_data_structure():

    print("Creating data structure for training...")
    if not os.path.exists(DATA_MOCK_DIR):
        os.makedirs(DATA_MOCK_DIR)
    
    for category in CLASS_NAMES:
        path = os.path.join(DATA_MOCK_DIR, category)
        os.makedirs(path, exist_ok=True)
        
    
    print("Data Structure created")

def build_cnn_model(input_shape, num_classes):
    """Defines the CNN architecture for land use classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def generate_synthetic_data(num_samples, img_size, num_classes):
    """Generates synthetic image data and labels to allow the script to run without a large dataset."""
    X = np.random.rand(num_samples, img_size[0], img_size[1], 3).astype('float32')
    y = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, num_samples), num_classes=num_classes)
    return X, y

def train_model():
    """Trains the model using mock data and saves the result."""
    
    # 1. Create Model
    model = build_cnn_model(IMG_SIZE + (3,), NUM_CLASSES)
    model.summary()

    # 2. Generate Synthetic Data 
    # in your own setup use your dataset
    # using flow_from_directory on your real satellite image folders.
    X_train, y_train = generate_synthetic_data(100, IMG_SIZE, NUM_CLASSES)
    
    # 3. Define Checkpoint
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', mode='min')
    
    # 4. Train the Model (using a small validation split on synthetic data)
    print("\nStarting model training with synthetic data...")
    try:
        model.fit(
            X_train, 
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2, # Use 20% of synthetic data for validation
            callbacks=[checkpoint],
            verbose=1
        )
        print(f"\nTraining complete. Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"\nAn error occurred during training (likely due to synthetic data setup): {e}")
        print("Saving a dummy model to ensure 'classify_image.py' can run.")
        # Save the untrained model if training failed (to unblock classification script)
        model.save(MODEL_PATH)

if __name__ == '__main__':
    train_model()