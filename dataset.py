import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class SuperResolutionDataset(Dataset):
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        """
        低画質（LR）と高画質（HR）の画像をペアで読み込むデータセット

        Args:
            low_res_dir (str): 低画質画像のフォルダパス
            high_res_dir (str): 高画質画像のフォルダパス
            transform (callable, optional): 入力画像に適用する変換（反転など）
        """
        self.low_res_images = sorted(glob(os.path.join(low_res_dir, "*.*")))
        self.high_res_images = sorted(glob(os.path.join(high_res_dir, "*.*")))

        self.transform_lr = transforms.Compose([
            transforms.Resize((256, 256)),
        ])

        self.transform_hr = transforms.Compose([
            transforms.Resize((512, 512)),
        ])

        self.transform = transform  # 引数で渡されたtransformを保存

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        """ LR（低画質）と HR（高画質）のペアを取得 """
        lr_image = Image.open(self.low_res_images[idx]).convert("RGB")
        hr_image = Image.open(self.high_res_images[idx]).convert("RGB")

        lr_image, hr_image = self.transform(lr_image, hr_image)

        lr_image = self.transform_lr(lr_image)
        hr_image = self.transform_hr(hr_image)

        return lr_image, hr_image
    
class PairedTransform:
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomChoice([
                transforms.Lambda(lambda img: img.rotate(0)),
                transforms.Lambda(lambda img: img.rotate(90, expand=True)),
                transforms.Lambda(lambda img: img.rotate(180)),
                transforms.Lambda(lambda img: img.rotate(270, expand=True)),
            ]),
            transforms.RandomResizedCrop(512, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __call__(self, lr_image, hr_image):
        seed = random.randint(0, 99999)  # 乱数のシードを固定
        random.seed(seed)
        torch.manual_seed(seed)
        lr_image = self.base_transform(lr_image)

        random.seed(seed)
        torch.manual_seed(seed)
        hr_image = self.base_transform(hr_image)

        return lr_image, hr_image
