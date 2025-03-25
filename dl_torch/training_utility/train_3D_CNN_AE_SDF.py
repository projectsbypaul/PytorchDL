import torch
import torch.nn as nn
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.models.CNN3D_Autoencoder import Autoencoder
from dl_torch.data_utility.DataAugmentation import random_flip_dataset
import os
from torch.utils.tensorboard import SummaryWriter

def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                criterion,
                model_save_name : str,
                device : torch.device,
                backup_epoch : int= 100,
                num_epochs=200,
                model_weights_loc : str = None
                ):

    # dataset properties
    print(f"Train on {len(train_loader.dataset)} samples, validate on {len(val_loader.dataset)} samples")

    model.to(device)  # push model to GPU

    writer = SummaryWriter()

    # Epoch
    for epoch in range(num_epochs):
        model.train() # set train mode
        train_loss, correct, total = 0, 0, 0 # reset metrics for epoch

        # Iteration
        for data, target in train_loader:
            target = target.unsqueeze(1)
            data, target = data.to(device), target.to(device) #push data and labels to GPU

            # Batch
            optimizer.zero_grad() # resets gradients -> PyTorch accumulates gradients by default
            output = model(data)  # prediction step

            loss = criterion(output, target) # calculate by predictions
            loss.backward() # Compute gradient
            optimizer.step()

            train_loss += loss.item() # add batch loss to epoch loss

        #output epoch results
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_loader):.8f}")

        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)

        #backup model
        if(epoch % backup_epoch) == 0 and model_weights_loc:

            dir_path = os.path.dirname(model_weights_loc.format(model_save_name))

            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)

            backup_name = model_save_name + f"_epoch_{epoch}"
            torch.save(model.state_dict(), model_weights_loc.format(backup_name))

        # Reduce learning rate
        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                target = target.unsqueeze(1)
                data, target = data.to(device), target.to(device)  #push data and labels to GPU
                output = model(data) # prediction step
                _, predicted = torch.max(output, 1) # get index of max value
                loss = criterion(output, target)  # calculate by predictions
                val_loss += loss.item()  # add batch loss to epoch loss


        print(f"Val_Loss: {val_loss / len(val_loader):.8f} | Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        writer.add_scalar("Loss/Val", val_loss / len(val_loader), epoch)

    writer.close()
    print('Finished Training')

    return model

def validate_model(model, val_loader, device):
    # Log device and dataset properties
    print("Using device:", device)
    print(f"Validate on {len(val_loader.dataset)} samples")

    model.to(device) # push model to gpu
    model.eval()  # Set to evaluation mode

    val_correct, val_total = 0, 0
    all_predictions, all_labels = [], []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            _, predicted = torch.max(output, 1)  # Get predicted class
            target = torch.argmax(target, dim=1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    val_acc = 100.0 * val_correct / val_total
    print(f"Accuracy of the model: {val_acc:.2f}%")

    return val_acc, all_predictions, all_labels

def training_routine():

    # Check GPU
    log_cuda_status()

    # Save data
    model_save_name = "CNN3D_Autoencoder"
    backup_epoch = 100
    dataset_loc = r"../../data/datasets/ModelNet10/ModelNet10_AE_SDF_32_train.torch"
    model_weights_loc = "../../data/model_weights/CNN3D_Autoencoder/{}.pth"

    # Hyper-parameters
    num_classes=10
    num_epochs = 1000
    batch_size = 64
    learning_rate = 0.00001

    # Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder()

    # Setup training environment
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)  # Reduce LR every 20 epochs
    criterion = nn.MSELoss()
    # Setup dataset for training
    dataset = InteractiveDataset.load_dataset(dataset_loc)
    dataset.data = random_flip_dataset(dataset.data, 0.8)
    dataset.set_split(0.95)
    dataset.split_dataset()# Augmentation

    test_loader = dataset.get_test_loader(batch_size)
    train_loader = dataset.get_train_loader(batch_size)


    # Train model
    model = train_model(model,
                        train_loader,
                        test_loader,
                        optimizer,
                        scheduler,
                        criterion,
                        model_save_name,
                        device,
                        backup_epoch,
                        num_epochs,
                        model_weights_loc
                        )

    # Save model
    torch.save(model.state_dict(), model_weights_loc.format(model_save_name))

def validation_routine():

    #Check GPU
    log_cuda_status()

    # Save data
    model_save_name = "CNN3D_Autoencoder"
    dataset_loc = r"../../data/datasets/ModelNet10/ModelNet10_SDF_32_train.torch"
    model_weights_loc = f"../../data/model_weights/{model_save_name}/{model_save_name}.pth"


    # Hyper-parameters for model recreation
    num_classes = 10
    batch_size = 32

    # Setup model
    loaded_model = Autoencoder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model weights
    state_dict = torch.load(model_weights_loc, map_location=device)
    loaded_model.load_state_dict(state_dict)

    # push to GPU and set evaluation mode
    loaded_model.to(device)
    loaded_model.eval()


    # Setup Validation
    dataset = InteractiveDataset.load_dataset(dataset_loc)
    dataset.set_split(0.99)
    dataset.split_dataset()
    data_loader = dataset.get_train_loader(batch_size)

    # Do validation
    val_acc, predictions, labels = validate_model(loaded_model, data_loader, device)

    # print(f"Accuracy of the model: {val_acc:.2f}% on {len(predictions)} samples")

def log_cuda_status():
    print(f"CUDA available: {torch.cuda.is_available()}")  # True if a GPU is available
    print(f"Number available GPUs {torch.cuda.device_count()}")  # Number of GPUs
    print(f"GPU index {torch.cuda.current_device()}")  # Current GPU index
    print(torch.cuda.get_device_name(0))
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

def main(mode: int):
    if mode==0:
      training_routine()
    elif mode==1:
      validation_routine()

if __name__ == "__main__":
    main(0)