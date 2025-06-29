from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import tensorflow as tf # type: ignore

# Create ImageDataGenerator with MobileNetV2 preprocessing
image_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

# Train generator
train = image_gen.flow_from_dataframe(
    dataframe=train_set, # type: ignore
    x_col="filepaths",
    y_col="labels",
    target_size=(244, 244),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=8,
    shuffle=False
)

# Test generator
test = image_gen.flow_from_dataframe(
    dataframe=test_images, # type: ignore
    x_col="filepaths",
    y_col="labels",
    target_size=(244, 244),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=8,
    shuffle=False
)

# Validation generator
val = image_gen.flow_from_dataframe(
    dataframe=val_set, # type: ignore
    x_col="filepaths",
    y_col="labels",
    target_size=(244, 244),
    color_mode='rgb',
    class_mode="categorical",
    batch_size=8,
    shuffle=False
)
