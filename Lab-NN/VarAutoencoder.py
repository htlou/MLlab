import torch
from torch import nn
from torch.nn import functional as F

VAE_ENCODING_DIM = 64
IMG_WIDTH, IMG_HEIGHT = 24, 24

# Define the Variational Encoder
class VarEncoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarEncoder, self).__init__()
        # TODO: implement the encoder
        self.fc1 = nn.Linear(3 * IMG_WIDTH * IMG_HEIGHT, 512)
        self.fc_mu = nn.Linear(512, encoding_dim)
        self.fc_log_var = nn.Linear(512, encoding_dim)


    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        
        # TODO: implement the forward pass
        x = torch.flatten(x, start_dim=1)
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

# Define the Decoder
class VarDecoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarDecoder, self).__init__()
        # TODO: implement the decoder
        self.fc = nn.Linear(encoding_dim, 3 * IMG_WIDTH * IMG_HEIGHT)


    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        
        # TODO: implement the forward pass
        h = F.relu(self.fc(v))
        x = h.view(-1, 3, IMG_WIDTH, IMG_HEIGHT)
        return x

# Define the Variational Autoencoder
class VarAutoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarAutoencoder, self).__init__()
        self.encoder = VarEncoder(encoding_dim)
        self.decoder = VarDecoder(encoding_dim)

    @property
    def name(self):
        return "VAE"

    def reparameterize(self, mu, log_var):
        '''
        mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        return v: sampled latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        
        # TODO: implement the reparameterization trick to sample v
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
        
    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        # TODO: implement the forward pass
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var

# Loss Function
def VAE_loss_function(outputs, images):
    '''
    outputs: (x, mu, log_var)
    images: input/original images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
    return loss: the loss value, dim: (1)
    '''
    # TODO: implement the loss function for VAE
    x_reconstructed, mu, log_var = outputs
    # Reconstruction loss (e.g., MSE)
    recon_loss = F.mse_loss(x_reconstructed, images, reduction='sum')
    # KL divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kld_loss


