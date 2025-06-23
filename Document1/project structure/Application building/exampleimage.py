import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import os
# --- Paths ---
train_dir = 'split_data/train'
val_dir = 'split_data/val'
# --- Image settings ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32




# --- Load images ---
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'  )
# --- Load pre-trained model ---
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False  # freeze layers
# --- Add custom layers ---
x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
# --- Compile ---
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# --- Train ---
model.fit(train_gen, validation_data=val_gen, epochs=5)
# --- Save model---
model.save("Blood_cells.h5")
print("✅ Model trained and saved as Blood_cells.h5")
