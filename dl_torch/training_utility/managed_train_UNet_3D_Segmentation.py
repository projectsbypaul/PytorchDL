import torch
import torch.nn as nn
from dl_torch.model_utility import Custom_Metrics
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL
import os
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from dl_torch.model_utility.Scheduler import get_linear_scheduler
from dl_torch.data_utility.InteractiveDatasetManager import InteractiveDatasetManager

def train_model(model,
                dataset_manager,
                optimizer,
                scheduler,
                criterion,
                model_name: str,
                device: torch.device,
                backup_epoch: int = 100,
                num_epochs=200,
                model_weights_loc: str = None
                ):

    train_size, val_size = dataset_manager.get_train_test_size()

    print(f"Train on {dataset_manager.get_subset_count()} Managed Dataset samples")
    print(f"Training samples {train_size}, Validation samples {val_size} ")


    model.to(device)
    run_name = (f"{model_name}"
                f"_lr[{scheduler.get_last_lr()[0]}]"
                f"_lrdc[{scheduler.final_lr / scheduler.get_last_lr()[0]:.0e}]"
                f"_bs{dataset_manager.get_batch_size()}")
    print(f"Model: {run_name}")

    writer = SummaryWriter(f"../dl_torch/training_utility/runs/{run_name}")

    for epoch in range(num_epochs):

        epoch_train_loss, epoch_train_acc = 0.0, 0.0
        epoch_val_loss, epoch_val_acc = 0.0, 0.0
        train_batches, val_batches = 0, 0

        for set_index in range(dataset_manager.get_subset_count()):
            dataset_manager.activate_subset_by_index(set_index)

            train_loader = dataset_manager.get_active_train_loader()
            val_loader = dataset_manager.get_active_test_loader()

            print(f"Epoch {epoch}: Loaded subset {set_index}, training...")
            model.train()

            for data, target in train_loader:
                data = data.to(device)
                target = torch.argmax(target, dim=1).long().to(device)  # [B, D, H, W]

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                epoch_train_acc += Custom_Metrics.voxel_accuracy(output, target)
                train_batches += 1

            print(f"Epoch {epoch}: Run validation on subset {set_index}...")
            model.eval()

            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(device)
                    target = torch.argmax(target, dim=1).long().to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    epoch_val_loss += loss.item()
                    epoch_val_acc += Custom_Metrics.voxel_accuracy(output, target)
                    val_batches += 1

        # Compute averages
        avg_train_loss = epoch_train_loss / max(train_batches, 1)
        avg_train_acc = epoch_train_acc / max(train_batches, 1)
        avg_val_loss = epoch_val_loss / max(val_batches, 1)
        avg_val_acc = epoch_val_acc / max(val_batches, 1)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.8f} | Train Acc: {avg_train_acc:.4%}")
        print(f"Val Loss: {avg_val_loss:.8f} | Val Acc: {avg_val_acc:.4%} | LR: {scheduler.get_last_lr()[0]:.4e}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Val", avg_val_acc, epoch)

        torch.cuda.empty_cache()

        if (epoch % backup_epoch) == 0 and model_weights_loc:
            backup_name = model_weights_loc.format(model_name=model_name, run_name=run_name, epoch=epoch)
            dir_path = os.path.dirname(backup_name)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(model.state_dict(), backup_name)

        scheduler.step()

    writer.close()
    print('Finished Training')
    return model

def training_routine(model_name : str,
                     dataset_dir : str,
                     model_weights_loc : str,
                     epochs : int,
                     backup_epochs : int,
                     batch_size : int,
                     lr,
                     decay_order,
                     split
                     ):
    #setup model and data
    model_constructor = partial(UNet3D_16EL, in_channels=1, out_channels=10)
    dataset_manager = InteractiveDatasetManager(dataset_dir, split, batch_size)

    # Check GPU
    log_cuda_status()

    # Setup Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup training environment
    criterion = nn.CrossEntropyLoss()

    # Training Env
    model = model_constructor()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_linear_scheduler(optimizer, lr, lr * decay_order, epochs)

    # Train loop
    model = train_model(model,
                        dataset_manager,
                        optimizer,
                        scheduler,
                        criterion,
                        model_name,
                        device,
                        backup_epochs,
                        epochs,
                        model_weights_loc
                        )

    run_name = f"{model_name}_lr[{lr}]_lrdc[{decay_order}]bs{batch_size}"

    model_save_name = model_weights_loc.format(model_name=model_name, run_name=run_name, epoch="last")

    # Save model
    torch.save(model.state_dict(), model_save_name)

def log_cuda_status():
    print(f"CUDA available: {torch.cuda.is_available()}")  # True if a GPU is available
    print(f"Number available GPUs {torch.cuda.device_count()}")  # Number of GPUs
    print(f"GPU index {torch.cuda.current_device()}")  # Current GPU index
    print(torch.cuda.get_device_name(0))
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

def main():
    print(torch.__version__)
    model_name = "UNet3D_SDF_16EL_n_class_10_multiset"
    dataset_dir = r"H:\ABC\ABC_torch\ABC_chunk_00\batched_data_ks_16_pad_4_bw_5_vs_adaptive_n2_testing"
    model_weights_loc = "../../data/model_weights/{model_name}/{run_name}_save_{epoch}.pth"

    epochs = 200
    backup_epochs = 100
    learning_rate = 1e-4
    decay_order = 1e-1
    batch_size = 4
    split = 0.9

    training_routine(model_name, dataset_dir, model_weights_loc, epochs, backup_epochs, batch_size, learning_rate, decay_order, split)

if __name__ == "__main__":
  main()