import torch
from model import Generator
from PIL import Image
import torchvision.transforms as transforms

class Upscaler():
    def __init__(self):
        self.generator = Generator().cuda()
        # 画像を読み込む
        self.num = 4
        self.image = Image.open("w_test" + str(self.num) +".png")
        self.original_width, self.original_height = self.image.size

    def upscale(self, batch_idx):
        self.generator.load_state_dict(torch.load("tmp_generator/generator_batch_" + str(batch_idx) + ".pth", weights_only=True))
        self.generator.eval()

        # 画像の前処理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image = transform(self.image).unsqueeze(0).cuda()

        # 画像を生成
        with torch.no_grad():
            fake_hr = self.generator(image)

        # 生成された画像を保存
        fake_hr = fake_hr.squeeze(0).cpu()
        fake_hr = transforms.ToPILImage()(fake_hr)
        fake_hr = fake_hr.resize((self.original_width * 2, self.original_height * 2), Image.BICUBIC)
        fake_hr.save("w_output" + str(self.num) +".jpg")
