import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PairedTransform, SuperResolutionDataset
from model import Generator, Discriminator
import os
from torchvision.utils import save_image
from upscaler import Upscaler

# ハイパーパラメータ
epochs = 4  # 学習回数
batch_size = 20  # バッチサイズ（GPUのメモリに依存）
lr_g = 1.0e-5  # Generatorの学習率
lr_d = 1.0e-7  # Discriminatorの学習率
 
# データセットの作成
transform = PairedTransform()
dataset = SuperResolutionDataset(low_res_dir="data/train_low", high_res_dir="data/train_high", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルのインスタンス化
generator = Generator().cuda()
discriminator = Discriminator(img_size=512).cuda()

# オプティマイザーの設定
optim_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))

# 損失関数
criterion = torch.nn.BCEWithLogitsLoss() 

# 途中から再開するための変数
start_epoch = 1  # デフォルトは1から開始

# アップスケーラー
upscaler = Upscaler()

# もしチェックポイントファイルが存在すれば、読み込む
checkpoint_path = "checkpoint/checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # 次のエポックから開始
    print(f"✅ 学習を {start_epoch} エポック目から再開します。")
else:
    print("🚀 チェックポイントも学習済みモデルも見つかりませんでした。新規学習を開始します。")

# 学習ループ
for epoch in range(start_epoch, epochs + 1):
    for batch_idx, (lr, hr) in enumerate(dataloader):
        # データをGPUに送る
        lr, hr = lr.cuda(), hr.cuda()

        # -----------------
        # Discriminatorの訓練
        # -----------------
        optim_d.zero_grad()

        # 本物の画像と生成された画像
        real_output = discriminator(hr)
        fake_hr = generator(lr)
        fake_output = discriminator(fake_hr.detach())  # 学習をバックプロパゲーションしない
        
        # 本物と偽物の判定損失
        real_loss = criterion(real_output, torch.ones_like(real_output))
        fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optim_d.step()

        # -----------------
        # Generatorの訓練
        # -----------------
        optim_g.zero_grad()

        # Generatorの出力をDiscriminatorに通して、損失を計算
        fake_output = discriminator(fake_hr)
        g_loss = criterion(fake_output, torch.ones_like(fake_output))

        g_loss.backward()
        optim_g.step()

        save_image(lr, f"real_low_images/batch_{batch_idx}.png", normalize=True)
        save_image(fake_hr, f"fake_images/batch_{batch_idx}.png", normalize=True)
        save_image(hr, f"real_images/batch_{batch_idx}.png", normalize=True)

        # ログの表記
        if batch_idx % 2 == 0:
            print(f"{epoch},{batch_idx + 1},{d_loss.item()},{g_loss.item()}")

        # キャッシュ
        if batch_idx % 2 == 0:
            torch.save(generator.state_dict(), f"tmp_generator/generator_batch_{batch_idx}.pth")
            upscaler.upscale(batch_idx)

    
    # nエポックごとにモデルを保存(状況に応じてn>0を設定してください)
    if epoch % 1 == 0:
        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
        }, f"checkpoint/checkpoint_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
        }, "checkpoint/checkpoint.pth")
        print(f"💾 チェックポイントを保存しました（Epoch {epoch}）")
