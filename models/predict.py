import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# 1. Load the fine-tuned model
MODEL_PATH = 'waste_classifier_finetuned.h5'
if not os.path.exists(MODEL_PATH):
    print(f"Error: {MODEL_PATH} not found. Using the base model instead.")
    MODEL_PATH = 'waste_classifier.h5'

model = tf.keras.models.load_model(MODEL_PATH)

# 2. Define classes (Ensure alphabetical order to match training)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict_waste(img_path):
    print(f"\nAnalyzing image: {img_path}...")
    
    # Load and resize image to 224x224 (as per our training config)
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert to array and scale pixels (1./255)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Get prediction probabilities
    predictions = model.predict(img_array)
    score = predictions[0]
    
    # Identify top result
    result_index = np.argmax(score)
    result_label = class_names[result_index]
    confidence = 100 * score[result_index]

    # Display results
    print("=" * 30)
    print(f"SMART CITY WASTE DETECTOR")
    print("=" * 30)
    print(f"PREDICTION : {result_label.upper()}")
    print(f"CONFIDENCE : {confidence:.2f}%")
    print("-" * 30)
    
    # Show how the AI "thought" about the other categories
    print("Probability Breakdown:")
    for i in range(len(class_names)):
        print(f" - {class_names[i]:10}: {100 * score[i]:5.2f}%")

# 3. Execution
if __name__ == "__main__":
    test_img = 'test.jpg' # Change this to your photo's name
    if os.path.exists(test_img):
        predict_waste(test_img)
    else:
        print(f"File {test_img} not found. Please place an image in the folder.")