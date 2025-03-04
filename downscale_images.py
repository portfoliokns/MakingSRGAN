import os
from glob import glob
from PIL import Image
import shutil

input_folder = "C:/Users/ransu/Downloads/train/old3"
output_folder_low = "data/train_low"
output_folder_high = "data/train_high"

os.makedirs(output_folder_high, exist_ok=True)
os.makedirs(output_folder_low, exist_ok=True)

image_files = glob(os.path.join(input_folder, "*.*"))

for image_path in image_files:
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width >= 1024 and height >= 1024:
                new_size = (width // 2, height // 2)
                img_resized = img.resize(new_size, Image.BICUBIC)

                filename = os.path.basename(image_path)
                save_path_low = os.path.join(output_folder_low, filename)
                img_resized.save(save_path_low)
                print(f"保存: {save_path_low}")

                # 元画像を high に移動
                save_path_high = os.path.join(output_folder_high, filename)
                shutil.move(image_path, save_path_high)
                print(f"移動: {save_path_high}")
            # else:
                # print(f"スキップ (サイズが小さいため): {image_path}")

    except Exception as e:
        print(f"Error {image_path}: {e}")

print("Finish Resize For All Images!")