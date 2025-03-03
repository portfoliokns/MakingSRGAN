import torch
from model import Generator
from PIL import Image
import torchvision.transforms as transforms

class Upscaler():
    def __init__(self):
        self.generator = Generator().cuda()
        # 画像を読み込む
        self.num = 4
        self.image = Image.open("w_test" + str(self.num) +".png")  # テスト用低解像度画像を指定
        self.original_width, self.original_height = self.image.size

    def upscale(self, batch_idx):
        self.generator.load_state_dict(torch.load("tmp_generator/generator_batch_" + str(batch_idx) + ".pth", weights_only=True))
        self.generator.eval()  # 評価モードに設定

        # # RGBA → RGBに変換（もし画像がRGBAであれば）
        # if self.image.mode != 'RGB':
        #     self.image = self.image.convert('RGB')

        # 画像の前処理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),  # 低解像度画像のリサイズ
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image = transform(self.image).unsqueeze(0).cuda()  # バッチ次元を追加し、GPUに送る

        # 画像を生成
        with torch.no_grad():
            fake_hr = self.generator(image)

        # 生成された画像を保存
        fake_hr = fake_hr.squeeze(0).cpu()  # バッチ次元を削除し、CPUに戻す
        fake_hr = transforms.ToPILImage()(fake_hr)  # TensorからPILに変換

        fake_hr = fake_hr.resize((self.original_width * 2, self.original_height * 2), Image.BICUBIC)
        fake_hr.save("w_output" + str(self.num) +".jpg")  # 保存
