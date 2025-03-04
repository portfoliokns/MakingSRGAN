import torch
from model import Generator
from PIL import Image
import torchvision.transforms as transforms

# 学習済みモデルの読み込み
generator = Generator().cuda()
generator.load_state_dict(torch.load("tmp_generator/generator_batch_520.pth", weights_only=True))
generator.eval()  # 評価モードに設定

# 画像を読み込む
num = 2
image = Image.open("w_test" + str(num) +".png")  # テスト用低解像度画像を指定

original_width, original_height = image.size

# 画像の前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
image = transform(image).unsqueeze(0).cuda()

# 画像を生成
with torch.no_grad():
    fake_hr = generator(image)

# 生成された画像を保存
fake_hr = fake_hr.squeeze(0).cpu()
fake_hr = transforms.ToPILImage()(fake_hr)

fake_hr = fake_hr.resize((original_width * 2, original_height * 2), Image.BICUBIC)
fake_hr.save("w_output" + str(num) +".jpg")
