import torch
import torch.nn as nn
import torch.nn.functional as F

AE_ENCODING_DIM = 64
H, W = 24, 24

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        self.encoding_dim = encoding_dim
        self.fc = None  
        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)




    def forward(self, v):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return v: latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        if self.fc is None:
            # 假设潜在向量应该展开成一个平方形状的特征图
            feature_size = int((self.encoding_dim // 128) ** 0.5)
            self.fc = nn.Linear(self.encoding_dim, 128 * feature_size * feature_size).to(v.device)

        # 从潜在向量生成特征图
        x = self.fc(v)
        feature_size = int(x.size(-1) ** 0.5)
        x = x.view(x.size(0), 128, feature_size, feature_size)

        # 应用转置卷积层和ReLU激活函数
        x = F.relu(self.conv_trans1(x))
        x = F.relu(self.conv_trans2(x))
        x = torch.sigmoid(self.conv_trans3(x))  # 使用sigmoid确保输出在[0, 1]范围内

        return x


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        # TODO: implement the decoder
        self.fc = nn.Linear(encoding_dim, 128 * H/4 * W/4)

        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)




    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        
        
        # TODO: implement the forward pass
        x = self.fc(v)
        x = x.view(x.size(0), 128, H//4, W//4)
        x = F.relu(self.conv_trans1(x))
        x = F.relu(self.conv_trans2(x))
        x = torch.sigmoid(self.conv_trans3(x))  
        return x


# Combine the Encoder and Decoder to make the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoding_dim)
        self.decoder = Decoder(encoding_dim)

    def forward(self, x):
        v = self.encoder(x)
        x = self.decoder(v)
        return x
    
    @property
    def name(self):
        return "AE"

