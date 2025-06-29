import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_data = datagen.flow_from_directory(
    r"C:\Users\kurma\Downloads\Rice_Image_Dataset",
    target_size=(96, 96),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)
train_data.samples = 1000
val_data = datagen.flow_from_directory(
    r"C:\Users\kurma\Downloads\Rice_Image_Dataset",
    target_size=(96, 96),
    batch_size=16,
    class_mode='sparse',
    subset='validation'
)

mobilenet_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5"
feature_extractor_layer = hub.KerasLayer(mobilenet_url, input_shape=(96, 96, 3), trainable=False)

num_classes = len(train_data.class_indices)

model = keras.Sequential([
    feature_extractor_layer,
    layers.Dense(num_classes)  # logits
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_data, validation_data=val_data, epochs=1)

model.save("model.h5")
print("✅ model.h5 saved successfully.")
