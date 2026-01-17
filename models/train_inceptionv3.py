import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, applications
from tensorflow.keras.applications import InceptionV3

# Hide GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Configuration
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = "../data" 

# 2. Data Loading (The "Old Reliable" Way)
# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,      # Rotate images up to 40 degrees
    width_shift_range=0.2,  # Shift images horizontally
    height_shift_range=0.2, # Shift images vertically
    shear_range=0.2,        # Distort images
    zoom_range=0.2,         # Zoom in/out
    horizontal_flip=True,   # Flip images
    fill_mode='nearest'     # Fill in new pixels after rotation
)

train_ds = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    seed=123
)

val_ds = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    seed=123
)

# 3. Build Model using Transfer Learning (MobileNetV2)
# We use a smaller alpha (0.35) to make it even lighter for CPU training
base_model = InceptionV3(
    input_shape=(224, 224, 3), # Note: Inception prefers 299x299
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(6, activation='softmax') # 6 classes: cardboard, glass, etc.
])

# 4. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train
print("\nStarting Training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# 1. Unfreeze the base model
base_model.trainable = True

# 2. Refreeze everything EXCEPT the last 20 layers
# This prevents destroying the well-learned features in the early layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

# 3. Re-compile with a VERY small learning rate
# High learning rates will destroy the pre-trained weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5), # 0.00001
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nStarting Fine-Tuning...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

model.save("waste_classifier_finetuned.h5")

# 6. Save
model.save("waste_classifier.h5")
print("\nModel saved successfully as waste_classifier.h5")