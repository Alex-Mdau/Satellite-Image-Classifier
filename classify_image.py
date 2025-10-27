import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


IMG_SIZE = (64, 64)
MODEL_PATH = 'data/model/land_use_classifier.h5'
CLASS_NAMES = ['Agriculture', 'Forest', 'Urban', 'Water']

def preprocess_image(image_path):
    """Loads, resizes, and normalizes an image for model prediction."""
    try:
        # Load the image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to the model's required input size
        img = img.resize(IMG_SIZE)
        
        # Convert to numpy array
        img_array = img_to_array(img)
        
        # Add batch dimension and normalize (expected by model)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        return img_array

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

def classify_image(image_path):
    
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model. Did you run 'python train_model.py' first? Error: {e}")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    input_tensor = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(input_tensor)[0]
    
    # Get the index of the highest probability
    predicted_index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = predictions[predicted_index] * 100
    
    # --- Output ---
    print("-" * 40)
    print(f"Classification Result for: {os.path.basename(image_path)}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 40)
    
    # Optionally print all class probabilities
    print("All Probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {predictions[i]*100:.2f}%")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python classify_image.py <path_to_image>")
        sys.exit(1)
        
    image_file = sys.argv[1]
    classify_image(image_file)