import os
import numpy as np
import disable_tensorflow_logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob

def count_files_in_folder(folder_path):
    file_pattern = os.path.join(folder_path, '*')  # Specify the pattern to match files in the folder
    file_paths = glob.glob(file_pattern)  # Get a list of file paths that match the pattern
    file_count = len(file_paths)  # Count the number of files

    return file_count

# Define the GAN architecture
latent_dim = 128

generator = keras.Sequential([
    layers.Dense(64 * 64 * 256, input_dim=latent_dim),
    layers.Reshape((64, 64, 256)),
    layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', activation='relu'),
    layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', activation='swish'),
    ##layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', activation='swish'),
    ##layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', activation='swish'),
    layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', activation='sigmoid')
], name='generator')

discriminator = keras.Sequential([
    layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[512, 512, 3]),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),
    layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),
    layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
    layers.LeakyReLU(0.2),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(1)
], name='discriminator')

gan = keras.Sequential([generator, discriminator])
discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                      loss=keras.losses.BinaryCrossentropy(from_logits=True))
gan.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss=keras.losses.BinaryCrossentropy(from_logits=True))

# Load and preprocess the dataset
image_folder = '/Volumes/Elements/GitHub/cats_with_birds/Torchy/calico_cat'  
# Replace with the path to your image folder

file_count = count_files_in_folder(image_folder)
print()
print(f"Number of files in the folder: {file_count}")
print()

image_files = [file for file in os.listdir(image_folder) if file.endswith('.jpg') or file.endswith('.jpeg')]
images = []
for image_file in image_files:
    image = tf.keras.preprocessing.image.load_img(
        os.path.join(image_folder, image_file), target_size=(512, 512)
    )
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    images.append(image)
dataset = np.array(images)

# Define the training loop
batch_size = 3
epochs = 200
sample_interval = 10

num_to_generate = 2  # Number of images to generate
output_folder = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/gan_gened'  # Folder to save the generated images
os.makedirs(output_folder, exist_ok=True)

##output_folder = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/gened_closed_mouth_cat'  # Folder to save the generated images
##os.makedirs(output_folder, exist_ok=True)


for epoch in range(epochs):
    # Select a random batch of images
    idx = np.random.randint(0, dataset.shape[0], batch_size)
    real_images = dataset[idx]

    # Generate a batch of fake images
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_images = generator.predict(noise)

    # Train the discriminator
    discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    discriminator_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    discriminator_loss = 0.5 * (discriminator_loss_real + discriminator_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print the progress
    print()    
    print(f'Epoch {epoch+1} completed')
    if (epoch + 1) % sample_interval == 0:
        generated_images = generator.predict(np.random.normal(0, 1, (num_to_generate, latent_dim)))
        for i, generated_image in enumerate(generated_images):
            image_path = os.path.join(output_folder, f"generated_image_{epoch+1}_{i}.png")
            keras.preprocessing.image.save_img(image_path, generated_image)
    else:
        print()
        print("no image generated")