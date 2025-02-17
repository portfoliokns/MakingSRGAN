import torch
from model import Generator
from PIL import Image
import torchvision.transforms as transforms

# 学習済みモデルの読み込み
generator = Generator().cuda()
generator.load_state_dict(torch.load("generator/generator_final.pth"))
generator.eval()  # 評価モードに設定

# 画像を読み込む
image = Image.open("w_test2.png")  # テスト用低解像度画像を指定

# RGBA → RGBに変換（もし画像がRGBAであれば）
if image.mode != 'RGB':
    image = image.convert('RGB')

original_width, original_height = image.size

# 画像の前処理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),  # 低解像度画像のリサイズ
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
image = transform(image).unsqueeze(0).cuda()  # バッチ次元を追加し、GPUに送る

# 画像を生成
with torch.no_grad():
    fake_hr = generator(image)

# 生成された画像を保存
fake_hr = fake_hr.squeeze(0).cpu()  # バッチ次元を削除し、CPUに戻す
fake_hr = transforms.ToPILImage()(fake_hr)  # TensorからPILに変換

fake_hr = fake_hr.resize((original_width * 2, original_height * 2), Image.BICUBIC)
fake_hr.save("w_output.jpg")  # 保存
