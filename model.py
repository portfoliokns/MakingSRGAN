import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_channels=3):
        super(Generator, self).__init__()

        # 残差ブロック
        self.block1 = self.make_block(input_channels, 64, 9)   # 入力 3チャンネル（RGB）→ 64チャンネル
        self.block2 = self.make_block(64, 64, 3)
        self.block3 = self.make_block(64, 64, 3)
        self.block4 = self.make_block(64, 64, 3)

        # アップサンプリング層
        self.upconv1 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, output_padding=0)

        self.activation = nn.Tanh()
    
    def make_block(self, in_channels, out_channels, kernel_size):
        """ 畳み込み層の作成 """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        """ 順伝播 """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # アップサンプリングを行う
        x = self.upconv1(x)  # サイズが2倍になる

        x = self.activation(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_channels=3, img_size=512):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)

        # 入力画像のサイズに基づいてfc_in_featuresを計算
        self.fc_in_features = self._calculate_fc_in_features(img_size)

        self.fc = nn.Linear(self.fc_in_features, 1)

    def _calculate_fc_in_features(self, img_size):
        """
        畳み込み層を通過した後の画像サイズを計算し、fc層の入力特徴量の数を返す
        """
        # 画像が2倍ずつ縮小するので、画像サイズを4回2倍で割って最終的なサイズを得る
        size = img_size
        for _ in range(4):  # conv4まで
            size = size // 2
        return 512 * size * size

    def forward(self, x):
        """ 順伝播 """
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)

        # Flatten: バッチサイズを保持して、フラットに変換
        x = x.view(x.size(0), -1)  # x.size(0) はバッチサイズ
        x = self.fc(x)
        return x