import torch
from model import Generator
from PIL import Image
import torchvision.transforms as transforms

class Upscaler():
    def __init__(self):
        self.generator = Generator().cuda()
        self.generator.eval()
        self.generator.cuda()

        # 画像を読み込む
        self.num = 4
        self.image = Image.open("w_test" + str(self.num) +".png")
        self.original_width, self.original_height = self.image.size
        if self.image.mode == "RGBA":
            self.image = self.image.convert("RGB")

    # デノーマライズ用の関数
    def denormalize(self, tensor):
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).cpu()
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).cpu()
        return tensor * std + mean

    def upscale(self, epoch, batch_idx):
        self.generator.load_state_dict(torch.load(f"tmp_generator/generator_batch_{epoch}_{batch_idx}.pth"))

        # 画像の前処理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
         ])
        image = transform(self.image).unsqueeze(0).cuda()

        # 画像を生成
        with torch.no_grad():
            fake_hr = self.generator(image)

        # 生成された画像を保存
        fake_hr = fake_hr.squeeze(0).cpu()
        fake_hr = self.denormalize(fake_hr)  # デノーマライズを適用
        fake_hr = torch.clamp(fake_hr, 0, 1)  # 範囲を [0,1] にクリップ
        fake_hr = transforms.ToPILImage()(fake_hr)

        # fake_hr = transforms.Resize((self.original_height * 2, self.original_width * 2), interpolation=Image.BICUBIC)(fake_hr)
        fake_hr = fake_hr.resize((self.original_width * 2, self.original_height * 2), Image.BICUBIC)
        fake_hr.save("w_output" + str(self.num) +".jpg")
        fake_hr.save(f"sample_images/{epoch}_{batch_idx}.jpg")
