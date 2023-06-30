import tensorflow as tf
from tensorflow.keras import layers

# Define the CNN model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # Three output classes: cats, mouthy cats, not cats
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Preprocess and load the image data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/training/directory',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=validation_split,
    subset='training',
    seed=123
)

validation_data = tf.keras.preprocessing.image_dataset_from_directory(
    'path/to/training/directory',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    validation_split=validation_split,
    subset='validation',
    seed=123
)

# Train the model
model.fit(
    train_data,
    validation_data=validation_data,
    epochs=num_epochs
)

# Save the trained model
model.save('path/to/save/model')
