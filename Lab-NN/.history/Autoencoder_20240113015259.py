import torch
import torch.nn as nn
import torch.nn.functional as F

AE_ENCODING_DIM = 64
H, W = 24

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        # TODO: implement the encoder
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)   # 32 * H/2 * W/2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 64 * H/4 * W/4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.fc = nn.Linear(128 * H/4 * W/4, encoding_dim)



    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return v: latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        # TODO: implement the forward pass

        # 应用卷积层和ReLU激活函数
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 展平数据以适配全连接层
        x = x.view(x.size(0), -1)

        # 生成潜在向量
        v = self.fc(x)


        return v


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        # TODO: implement the decoder

    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        
        
        # TODO: implement the forward pass
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

