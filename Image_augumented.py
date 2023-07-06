from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import numpy as np


# Set the path to your cat images
cat_images_path = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/non_cat'

# Create an ImageDataGenerator instance for data augmentation
datagen = ImageDataGenerator(
    rotation_range=12,  # Randomly rotate images by 20 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by 20% of the image width
    height_shift_range=0.2,  # Randomly shift images vertically by 20% of the image height
    shear_range=0.2,  # Randomly apply shearing transformations
    zoom_range=0.2,  # Randomly zoom in and out of images
    horizontal_flip=False,  # Randomly flip images horizontally
    vertical_flip=False,  # Randomly flip images vertically
    fill_mode='nearest'  # Fill in any newly created pixels after transformation
)

# Load and preprocess the cat images
def load_images():
    cat_images = []
    for filename in os.listdir(cat_images_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(os.path.join(cat_images_path, filename))
            img = img.resize((256, 256))  # Resize the images to the desired size
            cat_images.append((img, filename))  # Append a tuple of image data and image name
    return cat_images

# Perform data augmentation on the cat images


def save_img(file_path, image):
    # Convert the NumPy array to uint8 and reshape if necessary
    image = np.uint8(image)
    if image.ndim == 3 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
        
    image_pil = Image.fromarray(image)  # Convert the NumPy array back to PIL Image
    image_pil.save(file_path)  # Save the image to the specified file path


def augment_images(images, save_path, save_prefix, save_format='jpg'):
    for i, (image, image_name) in enumerate(images):
        image_array = np.array(image)
        image_array = image_array.reshape((1,) + image_array.shape)  # Reshape to 4D array for generator
        original_name = os.path.splitext(image_name)[0]  # Extract the original filename without extension
        original_name = original_name.replace("_", "")  # Remove underscores from original filename
        k = 17
        for j in range(k):
            augmented_name = f"{original_name}_{j}.{save_format}"
            save_file_path = os.path.join(save_path, augmented_name)
            
            # Generate augmented image and save it
            aug_image = next(datagen.flow(image_array, batch_size=1))
            save_img(save_file_path, aug_image[0])

# Load the cat images
cat_images = load_images()

augmented_images_path = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Now_train/non_cats'

# Perform data augmentation and save the augmented images
cat_images = load_images()
augment_images(cat_images, augmented_images_path, save_prefix='aug')

