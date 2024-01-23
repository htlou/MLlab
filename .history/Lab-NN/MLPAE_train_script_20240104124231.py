from autograd.BaseGraph import Graph
from autograd.Nodes import relu, Linear, MSE, sigmoid
from utils.scheduler import exponential_decay
import numpy as np
from utils.data_processor import create_flower_dataloaders
import pickle
import torch


# set random seed
np.random.seed(0)
torch.manual_seed(0)

# Basic settings
data_root = "./flowers"
vis_root = "./vis"
model_save_path = "./model/Best_MLPAE.pkl"
batch_size = 16
num_epochs = 100
early_stopping_patience = 5
IMG_WIDTH, IMG_HEIGHT = 24, 24


scheduler = exponential_decay(initial_learning_rate=1, decay_rate=0.9, decay_epochs=5)
training_dataloader, validation_dataloader = create_flower_dataloaders(batch_size, data_root, IMG_WIDTH, IMG_HEIGHT)


model = Graph([
    # TODO: Please implement the MLP autoencoder model here (Both encoder and decoder).
    Linear(3 * IMG_WIDTH * IMG_HEIGHT, 256),
    relu(),
    Linear(256, 128),
    relu(),
    Linear(128, 64),
    relu(),
    Linear(64, 128),
    relu(),
    Linear(128, 256),
    relu(),
    Linear(256, 3 * IMG_WIDTH * IMG_HEIGHT),
    sigmoid()
]
)

loss_fn_node = MSE()


save_model_name = f"Best_MLPAE.pkl"
min_valid_loss = float('inf')
avg_train_loss = 10000.
avg_valid_loss = 10000.

for epoch in range(num_epochs):
    
    train_losses = []
    
    # Adjust the learning rate
    lr = scheduler(epoch)
    step_num = len(training_dataloader)
    
    # Training all batches
    for images, _ in training_dataloader:
        images = images.detach().numpy() # (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        model.flush()
        loss_fn_node.flush()
        # TODO: Please implement the training loop for a MLP autoencoder, which can perform forward, backward and update the parameters.
        images = images.reshape(images.shape[0], -1)

        output = model.forward(images)
        print(output)
        loss = loss_fn_node.forward(output, images)
        print(loss)
        
        model.backward(loss_fn_node.backward())
        model.optimstep(lr)
        
        train_losses.append(loss.item())
        
        # print(loss.item())
    avg_train_loss = sum(train_losses) / len(train_losses)
    # raise NotImplementedError
    
    # Validation every 3 epochs
    if epoch % 3 == 0:
        valid_losses = []
        for images, _ in validation_dataloader:
            images = images.detach().numpy() # (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
            # TODO: Please implement code which calculates the validation loss.
            images = images.reshape(images.shape[0], -1) #flatten
            reconstructed = model.forward(images)
            val_loss = loss_fn_node.forward(reconstructed, images)
            
            valid_losses.append(val_loss.item())
        avg_valid_loss = sum(valid_losses) / len(valid_losses)

        # TODO: Save model if better validation loss is achieved. Save the model by calling pickle.dump(model, open(model_save_path, 'wb')).
        if avg_valid_loss < min_valid_loss:
            min_valid_loss = avg_valid_loss
            early_stopping_counter = 0
            pickle.dump(model, open(model_save_path, 'wb'))
        else:
            early_stopping_counter += 1

        
        # TODO: Early stopping if validation loss does not decrease for <early_stopping_patience> validation checks.
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
        
    print(f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}")

