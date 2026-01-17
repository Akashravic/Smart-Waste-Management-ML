import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 1. Load the fine-tuned model
model = tf.keras.models.load_model('waste_classifier_finetuned.h5')

# 2. Get the validation data (without shuffle to match labels)
DATA_DIR = "../data"
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_ds = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    subset='validation',
    shuffle=False  # IMPORTANT for confusion matrix
)

# 3. Make Predictions
print("Evaluating model...")
predictions = model.predict(val_ds)
y_pred = np.argmax(predictions, axis=1)
y_true = val_ds.classes
class_names = list(val_ds.class_indices.keys())

# 4. Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: Smart Waste Classifier')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
print("Confusion Matrix saved as confusion_matrix.png")

# 5. Print Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))