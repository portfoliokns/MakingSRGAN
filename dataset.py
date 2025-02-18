import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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
            transforms.Resize((128, 128)),  # 360p → 128×128 にリサイズ
        ])

        self.transform_hr = transforms.Compose([
            transforms.Resize((512, 512)),  # 1080p → 512×512 にリサイズ
        ])

        self.transform = transform  # 引数で渡されたtransformを保存

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        """ LR（低画質）と HR（高画質）のペアを取得 """
        lr_image = Image.open(self.low_res_images[idx]).convert("RGB")
        hr_image = Image.open(self.high_res_images[idx]).convert("RGB")

        if self.transform:
            lr_image = self.transform(lr_image)  # 低解像度画像にtransformを適用(反転)
            hr_image = self.transform(hr_image)  # 高解像度画像にtransformを適用(反転)

        lr_image = self.transform_lr(lr_image)
        hr_image = self.transform_hr(hr_image)

        return lr_image, hr_image
