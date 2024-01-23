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
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Linear layer for encoding
        self.fc = nn.Linear(in_features=128 * (H//4) * (W//4), out_features=encoding_dim)


    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return v: latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten and pass through the linear layer to get the latent vector
        x = torch.flatten(x, start_dim=1)
        v = self.fc(x)
        return v


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        
        # Linear layer for decoding
        self.fc = nn.Linear(in_features=encoding_dim, out_features=128 * (H//4) * (W//4))

        # Transposed convolutional layers
        self.convtranspose1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.convtranspose2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtranspose3 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, v):
        # Pass the latent vector through the linear layer and reshape
        x = F.relu(self.fc(v))
        x = x.view(-1, 128, H//4, W//4)

        # Apply transposed convolutions
        x = F.relu(self.convtranspose1(x))
        x = F.relu(self.convtranspose2(x))
        x = torch.sigmoid(self.convtranspose3(x))  # Sigmoid activation to ensure output is in [0, 1]
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

