import os
from glob import glob
from PIL import Image

input_folder = "C:/Users/ransu/Downloads/train/n02113186"
output_folder = "data/train_low"

os.makedirs(output_folder, exist_ok=True)

image_files = glob(os.path.join(input_folder, "*.*"))

for image_path in image_files:
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            new_size = (width // 2, height // 2)
            img_resized = img.resize(new_size)

            filename = os.path.basename(image_path)
            save_path = os.path.join(output_folder, filename)

            img_resized.save(save_path)
            print(f"Saved: {save_path}")

    except Exception as e:
        print(f"Error {image_path}: {e}")

print("Finish Resize For All Images!")