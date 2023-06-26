
import os
from PIL import Image

input_folder = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/closed_mouth_cat/'
output_folder = '/Volumes/Elements/GitHub/cats_with_birds/For_Training/resized_closed_mouth_cat/'  # Replace with the path to your output folder
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    
    # Skip non-image files
    if not os.path.isfile(input_path) or filename.startswith('.'):
        continue

    try:
        # Open the image
        image = Image.open(input_path)

        # Resize the image to 256x256
        image = image.resize((256, 256))

        # Convert the image to JPEG format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save the image
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')
        image.save(output_path, format='JPEG')

        print(f"Processed: {filename}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

print("Image resizing and format conversion complete.")
