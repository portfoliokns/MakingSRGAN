import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SuperResolutionDataset
from model import Generator, Discriminator
import torch.nn.functional as F
import os

# ハイパーパラメータ
epochs = 10
batch_size = 24
lr_g = 3e-4  # Generatorの学習率
lr_d = 1e-7  # Discriminatorの学習率

# データセットの作成
dataset = SuperResolutionDataset(low_res_dir="data/low", high_res_dir="data/high")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルのインスタンス化
generator = Generator().cuda()
discriminator = Discriminator(img_size=512).cuda()

# オプティマイザーの設定
optim_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
optim_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))

# 損失関数
criterion = torch.nn.MSELoss()  # Adversarial Loss

# 途中から再開するための変数
start_epoch = 1  # デフォルトは1から開始

# もしチェックポイントファイルが存在すれば、読み込む
checkpoint_path = "checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    optim_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    optim_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
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
        g_loss = criterion(fake_output, torch.ones_like(fake_output))  # 本物と認識させる

        g_loss.backward()
        optim_g.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}], Step [{batch_idx}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
    
    # 5エポックごとにモデルを保存
    if epoch % 5 == 0:
        torch.save(generator.state_dict(), f"generator_epoch_{epoch}.pth")
        torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optim_g.state_dict(),
            "optimizer_d_state_dict": optim_d.state_dict(),
        }, "checkpoint.pth")
        print(f"💾 チェックポイントを保存しました（Epoch {epoch}）")

torch.save(generator.state_dict(), f"generator_final.pth")
torch.save(discriminator.state_dict(), f"discriminator_final.pth")