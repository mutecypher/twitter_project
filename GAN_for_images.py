import tensorflow as tf
from tensorflow.keras import layers
import disable_tensorflow_logging
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense


import os
import numpy as np
from PIL import Image

# Define the target size and number of color channels for the generated images
GENERATOR_TARGET_SIZE = (256, 256)
NUM_COLOR_CHANNELS = 3

# Generator model
# Generator model
# Generator model
def build_generator():
    model = tf.keras.Sequential()

    # Input layer
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Reshape to match the target size
    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)

    # Transpose convolutional layers
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Adjust the output shape to match the target size (256, 256)
    model.add(layers.Conv2DTranspose(NUM_COLOR_CHANNELS, (5, 5), strides=(4, 4), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, GENERATOR_TARGET_SIZE[0], GENERATOR_TARGET_SIZE[1], NUM_COLOR_CHANNELS)

    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()

    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(GENERATOR_TARGET_SIZE[0], GENERATOR_TARGET_SIZE[1], NUM_COLOR_CHANNELS)))
    
    ##model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(GENERATOR_TARGET_SIZE[0], GENERATOR_TARGET_SIZE[1], 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model


# Loss function for both generator and discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Create models
generator = build_generator()
input_shape = (GENERATOR_TARGET_SIZE[0], GENERATOR_TARGET_SIZE[1], NUM_COLOR_CHANNELS)  # Update input shape
discriminator = build_discriminator()

# Training loop
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        resized_images = tf.image.resize(images, (GENERATOR_TARGET_SIZE[0], GENERATOR_TARGET_SIZE[1]))
        real_output = discriminator(resized_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# Set the path to your cat images
cat_images_path = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/resized_closed_mouth_cat'

# Load and preprocess the cat images
# Load and preprocess the cat images
def load_images():
    cat_images = []
    for filename in os.listdir(cat_images_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(cat_images_path, filename))
            img = img.resize(GENERATOR_TARGET_SIZE)  # Resize the images to match the GAN input size
            img = np.array(img.convert('L'))  # Convert to grayscale
            img = img.reshape((*GENERATOR_TARGET_SIZE, 1))  # Reshape to match the GAN input shape
            cat_images.append(img)
    return np.array(cat_images)

# Normalize the cat images to the range [-1, 1]
def normalize_images(images):
    return (images.astype(np.float32) - 127.5) / 127.5

# Prepare the training data
cat_images = load_images()
normalized_cat_images = normalize_images(cat_images)

# Set the batch size and number of epochs for training
BATCH_SIZE = 256
EPOCHS = 50

# Training loop
for epoch in range(EPOCHS):
    for batch in range(len(normalized_cat_images) // BATCH_SIZE):
        # Select a random batch of cat images
        images_batch = normalized_cat_images[np.random.randint(0, len(normalized_cat_images), BATCH_SIZE)]

        # Train the GAN
        train_step(images_batch)
    print()    
    print(f'Epoch {epoch+1} completed')
    

# Set the path to the folder for saving the generated images
output_folder = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/GAN_images/'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Number of images to generate
num_generated_images = 100

# Generate new cat images
random_noise = np.random.normal(size=(num_generated_images, 100))
generated_images = generator(random_noise, training=False)

# Save the generated images
for i in range(num_generated_images):
    img = generated_images[i] * 127.5 + 127.5  # Denormalize the image
    img = img.numpy().astype(np.uint8).reshape((28, 28))
    img = Image.fromarray(img, mode='L')
    img.save(os.path.join(output_folder, f'generated_cat_{i+1}.png'))

print(f"{num_generated_images} images generated and saved in {output_folder}.")
