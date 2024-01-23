import torch
from VarAutoencoder import VarAutoencoder, VAE_ENCODING_DIM
from Autoencoder import Autoencoder, AE_ENCODING_DIM
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--model", type=str, default="VAE", choices=["VAE", "AE"])
args = parser.parse_args()

model_save_path = f"./model/Best_{args.model}.pth"
vis_root = "./vis"
model_class = VarAutoencoder if args.model == "VAE" else Autoencoder
ENCODING_DIM = AE_ENCODING_DIM if args.model == "AE" else VAE_ENCODING_DIM


model = model_class(
    encoding_dim=ENCODING_DIM
)
model.load_state_dict(torch.load(model_save_path))



# TODO: Generate random images. Please sample 10 latent vectors from a standard normal distribution, and feed them to the decoder to generate images
# Please ensure that the generated images are tensors named as random_images, with shape (10, 3, 24, 24)
# Set the model to evaluation mode
model.eval()

# Generate 10 random latent vectors from a standard normal distribution
random_latent_vectors = torch.randn(10, ENCODING_DIM)

# Move the latent vectors to the same device as the model
random_latent_vectors = random_latent_vectors.to(next(model.parameters()).device)

# Generate images using the decoder
with torch.no_grad():
    if args.model == "VAE":
        random_images = model.decoder(random_latent_vectors)
    else:
        random_images = model.decoder(random_latent_vectors)


# Save the 10 random images in one figure
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(10, 1))

for i in range(10):
    ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(np.transpose(random_images[i].detach().numpy(), (1, 2, 0)))
    
plt.savefig(f"{vis_root}/random_images_{args.model}.png")