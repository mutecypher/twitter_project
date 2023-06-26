import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import os

# Generator model
def build_generator(latent_dim):
    model = keras.Sequential([
        layers.Dense(128, input_dim=latent_dim),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(2048),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
       ## layers.Dense(4096),
       ## layers.BatchNormalization(),
       ## layers.LeakyReLU(alpha=0.2),
        layers.Dense(256 * 256 * 3, activation='sigmoid'),
        layers.Reshape((256, 256, 3))
    ])
    return model

# Discriminator model
def build_discriminator():
    model = keras.Sequential([
        layers.Flatten(input_shape=(256, 256, 3)),
       ## layers.Dense(8192),
      ##  layers.BatchNormalization(),
      ##  layers.LeakyReLU(alpha=0.2),       
        layers.Dense(4096),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(2048),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1)
    ])
    return model

# Training parameters
latent_dim = 256
num_epochs = 10
##batch_size = 32
learning_rate = 0.0001

# Create the generator and discriminator
generator = build_generator(latent_dim)
discriminator = build_discriminator()

# Loss function and optimizers
##cross_entropy = keras.losses.BinaryCrossentropy()
##wasserstein_loss = keras.losses.mean_squared_error(real_labels, real_predictions) - keras.losses.mean_squared_error(fake_labels, fake_predictions)
cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.97)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5, beta_2=0.97)

# Load and preprocess your dataset here
image_directory = "/Volumes/Elements/GitHub/cats_with_birds/For_Training/black_cat2"
image_size = (256, 256 )
batch_size = 1

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image


image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory)if file.endswith('.jpg') or file.endswith('.jpeg')]

num_images = len(image_files)

dataset = tf.data.Dataset.from_tensor_slices(image_files)
dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.shuffle(num_images).batch(batch_size)

# Training loop
for epoch in range(num_epochs):
    for batch_images in dataset:
        batch_size = tf.shape(batch_images)[0]
        real_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            generated_images = generator(tf.random.normal((batch_size, latent_dim)))
            fake_labels = tf.zeros((batch_size, 1))

            real_predictions = discriminator(batch_images)
            real_loss = cross_entropy(real_labels, real_predictions)

            fake_predictions = discriminator(generated_images)
            fake_loss = cross_entropy(fake_labels, fake_predictions)

            discriminator_loss = 0.5 * (real_loss + fake_loss)

        gradients = tape.gradient(discriminator_loss, discriminator.trainable_variables)
        # Apply gradient clipping to the discriminator gradients
        clipped_gradients = [tf.clip_by_value(g, -0.2, 0.2) for g in gradients]
        discriminator_optimizer.apply_gradients(zip(clipped_gradients, discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as tape:
            generated_images = generator(tf.random.normal((batch_size, latent_dim)))
            fake_labels = tf.zeros((batch_size, 1))

            predictions = discriminator(generated_images)
            generator_loss = cross_entropy(fake_labels, predictions)

        gradients = tape.gradient(generator_loss, generator.trainable_variables)
        # Apply gradient clipping to the generator gradients
        clipped_gradients = [tf.clip_by_value(g, -0.2, 0.2) for g in gradients]
        generator_optimizer.apply_gradients(zip(clipped_gradients, generator.trainable_variables))

    # Print training progress
    print(f"Epoch [{epoch + 1}/{num_epochs}], Generator Loss: {generator_loss:.4f}, Discriminator Loss: {discriminator_loss:.4f}")

    # Save generated images
    generated_images = generator(tf.random.normal((2, latent_dim)))
    generated_images = (generated_images * 0.5 + 0.5) * 255  # De-normalize images
    for i, image in enumerate(generated_images):
        image = tf.reshape(image, (256, 256, 3)).numpy()
        image = Image.fromarray(image.astype(np.uint8))
        image.save(f"/Volumes/Elements/GitHub/cats_with_birds/For_Training/gan_gened/gan_aug{epoch + 1}_{i + 1}.jpg")


    # Save generated images
    generated_images = generator(tf.random.normal((2, latent_dim)))
    generated_images = (generated_images * 0.5 + 0.5) * 255  # De-normalize images
    for i, image in enumerate(generated_images):
        image = tf.reshape(image, (256, 256, 3)).numpy()
        image = Image.fromarray(image.astype(np.uint8))
        image.save(f"/Volumes/Elements/GitHub/cats_with_birds/For_Training/gan_gened/gan_aug{epoch + 1}_{i + 1}.jpg")
