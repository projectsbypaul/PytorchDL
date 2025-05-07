import torch
import torch.nn as nn
from requests import delete

from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.models.UNet3D_Autoencoder import AutoEncoder3D
from dl_torch.data_utility.DataAugmentation import random_flip_dataset
import os
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from dl_torch.model_utility.Scheduler import get_linear_scheduler

def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                criterion,
                model_name : str,
                device : torch.device,
                backup_epoch : int= 100,
                num_epochs=200,
                model_weights_loc : str = None
                ):

    # dataset properties
    print(f"Train on {len(train_loader.dataset)} samples, validate on {len(val_loader.dataset)} samples")

    model.to(device)  # push model to GPU

    run_name = (f"{model_name}"
                f"_lr[{scheduler.get_last_lr()[0]}]"
                f"_lrdc[{scheduler.final_lr/scheduler.get_last_lr()[0]:.0e}]"
                f"_bs{train_loader.batch_size}")

    print(f"Model: {run_name}")

    writer = SummaryWriter(f"runs/{run_name}")

    # Epoch
    for epoch in range(num_epochs):
        model.train() # set train mode
        train_loss, correct, total = 0, 0, 0 # reset metrics for epoch


        # Iteration
        for data, target in train_loader:
            target = target
            data, target = data.to(device), target.to(device) #push data and labels to GPU

            # Batch
            optimizer.zero_grad() # resets gradients -> PyTorch accumulates gradients by default
            output = model(data)  # prediction step

            loss = criterion(output, target) # calculate by predictions
            loss.backward() # Compute gradient
            optimizer.step()

            train_loss += loss.item() # add batch loss to epoch loss#



        if epoch == 0:
            print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

        #output epoch results
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss / len(train_loader):.8f}")

        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)

        backup_name = model_weights_loc.format(model_name=model_name, run_name=run_name, epoch=epoch)

        #backup model
        if(epoch % backup_epoch) == 0 and model_weights_loc:

            dir_path = os.path.dirname(backup_name)

            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)

            torch.save(model.state_dict(), backup_name)

        # Reduce learning rate
        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)  #push data and labels to GPU
                output = model(data) # prediction step
                _, predicted = torch.max(output, 1) # get index of max value
                loss = criterion(output, target)  # calculate by predictions
                val_loss += loss.item()  # add batch loss to epoch loss


        print(f"Val_Loss: {val_loss / len(val_loader):.8f} | Learning Rate: {scheduler.get_last_lr()[0]:.4e}")
        writer.add_scalar("Loss/Val", val_loss / len(val_loader), epoch)

    writer.close()
    print('Finished Training')

    return model

def training_routine(model_constructor,
                     dataset,
                     model_weights_loc,
                     model_name,
                     epoch,
                     backup_epoch,
                     lr,
                     decay_order,
                     batch_size
                     ):

    # Check GPU
    log_cuda_status()

    # Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup training environment
    criterion = nn.BCEWithLogitsLoss()

    for l in lr:
        for dc_od in decay_order:
            for bs in batch_size:
                #define batch size on data set
                test_loader = dataset.get_test_loader(bs)
                train_loader = dataset.get_train_loader(bs)

                # Training Env
                model = model_constructor()


                optimizer = torch.optim.AdamW(model.parameters(), lr=l, weight_decay=1e-4)
                scheduler = get_linear_scheduler(optimizer, l
                                                 , l*dc_od, epoch)

                # Train loop
                model = train_model(model,
                                    train_loader,
                                    test_loader,
                                    optimizer,
                                    scheduler,
                                    criterion,
                                    model_name,
                                    device,
                                    backup_epoch,
                                    epoch,
                                    model_weights_loc
                                    )

                run_name = f"{model_name}_lr[{l}]cdod[{dc_od}]bs{bs}"

                model_save_name = model_weights_loc.format(model_name=model_name, run_name=run_name, epoch=epoch)

                # Save model
                torch.save(model.state_dict(), model_save_name)




def log_cuda_status():
    print(f"CUDA available: {torch.cuda.is_available()}")  # True if a GPU is available
    print(f"Number available GPUs {torch.cuda.device_count()}")  # Number of GPUs
    print(f"GPU index {torch.cuda.current_device()}")  # Current GPU index
    print(torch.cuda.get_device_name(0))
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

def job_1():
    # Model Setup
    model_constructor = partial(AutoEncoder3D) # pass handle for constructor
    model_name = "Autoencoder_UNEt"
    dataset_loc = r"../../data/datasets/ABC/AE_ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2.torch"
    model_weights_loc = "../../data/model_weights/{model_name}/{run_name}_save_{epoch}.pth"

    epochs = 500
    backup_epochs = 100
    learning_rates = [10 ** -5]
    decay_order = [10 ** -1]
    batch_sizes = [4]

    # Setup dataset for training
    dataset = InteractiveDataset.load_dataset(dataset_loc)
    # Augmentation
    dataset.set_split(0.8)
    # dataset.data = random_flip_dataset(dataset.data, 0.7)
    dataset.split_dataset()

    # Train model
    training_routine(
        model_constructor,
        dataset,
        model_weights_loc,
        model_name,
        epochs,
        backup_epochs,
        learning_rates,
        decay_order,
        batch_sizes
    )

def job_2():
    # Model Setup
    model_constructor = partial(AutoEncoder3D)  # pass handle for constructor
    model_name = "Autoencoder_UNEt"
    dataset_loc = r"../../data/datasets/ABC/AE_ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2.torch"
    model_weights_loc = "../../data/model_weights/{model_name}/{run_name}_save_{epoch}.pth"

    epochs = 500
    backup_epochs = 100
    learning_rates = [10 ** -6]
    decay_order = [10 ** -1]
    batch_sizes = [4]

    # Setup dataset for training
    dataset = InteractiveDataset.load_dataset(dataset_loc)
    # Augmentation
    dataset.set_split(0.8)
    # dataset.data = random_flip_dataset(dataset.data, 0.7)
    dataset.split_dataset()

    # Train model
    training_routine(
        model_constructor,
        dataset,
        model_weights_loc,
        model_name,
        epochs,
        backup_epochs,
        learning_rates,
        decay_order,
        batch_sizes
    )

def job_3():
    # Model Setup
    model_constructor = partial(AutoEncoder3D)  # pass handle for constructor
    model_name = "Autoencoder_UNEt"
    dataset_loc = r"../../data/datasets/ABC/AE_ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2.torch"
    model_weights_loc = "../../data/model_weights/{model_name}/{run_name}_save_{epoch}.pth"

    epochs = 500
    backup_epochs = 100
    learning_rates = [10 ** -6]
    decay_order = [10 ** 0]
    batch_sizes = [4]

    # Setup dataset for training
    dataset = InteractiveDataset.load_dataset(dataset_loc)
    # Augmentation
    dataset.set_split(0.8)
    # dataset.data = random_flip_dataset(dataset.data, 0.7)
    dataset.split_dataset()

    # Train model
    training_routine(
        model_constructor,
        dataset,
        model_weights_loc,
        model_name,
        epochs,
        backup_epochs,
        learning_rates,
        decay_order,
        batch_sizes
        )


def main():
    job_1()
    job_2()
    job_3()


if __name__ == "__main__":
  main()