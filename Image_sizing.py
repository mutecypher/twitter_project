from PIL import Image
import os

def get_image_dimensions(image_path):
    """
    Get the dimensions of the image at the given path.
    :param image_path: The path to the image file.
    :return: The image width and height.
    """
    image = Image.open(image_path)
    depth = image.mode
    return image.size, depth

    print("Image dimensions:")
    print(f"Width: {width}px")
    print(f"Height: {height}px")



image_folder = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/Augmented2/' 
image_files = os.listdir(image_folder)
images = []
for image_file in image_files:
    sized, deep = get_image_dimensions(os.path.join(image_folder, image_file))
    if sized != (256, 256) and deep != 'RGB':
        print(f"for file: {image_file} the width and height are: {sized} and the depth is {deep}")
