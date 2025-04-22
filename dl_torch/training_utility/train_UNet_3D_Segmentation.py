import torch
import torch.nn as nn
from dl_torch.model_utility import Custom_Metrics
from dl_torch.data_utility.InteractiveDataset import InteractiveDataset
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL
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
                model_name: str,
                device: torch.device,
                backup_epoch: int = 100,
                num_epochs=200,
                model_weights_loc: str = None
                ):

    print(f"Train on {len(train_loader.dataset)} samples, validate on {len(val_loader.dataset)} samples")

    model.to(device)
    run_name = (f"{model_name}"
                f"_lr[{scheduler.get_last_lr()[0]}]"
                f"_lrdc[{scheduler.final_lr / scheduler.get_last_lr()[0]:.0e}]"
                f"_bs{train_loader.batch_size}")
    print(f"Model: {run_name}")

    writer = SummaryWriter(f"runs/{run_name}")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for data, target in train_loader:
            data = data.to(device)

            target = torch.argmax(target, dim=1).long().to(device)  # [B, D, H, W]

            optimizer.zero_grad()
            output = model(data)  # [B, 7, D, H, W]
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += Custom_Metrics.voxel_accuracy(output, target)

        if epoch == 0:
            print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.8f} | Train Acc: {avg_train_acc:.4%}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)

        backup_name = model_weights_loc.format(model_name=model_name, run_name=run_name, epoch=epoch)

        if (epoch % backup_epoch) == 0 and model_weights_loc:
            dir_path = os.path.dirname(backup_name)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            torch.save(model.state_dict(), backup_name)

        scheduler.step()

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = torch.argmax(target, dim=1).long().to(device)  # [B, D, H, W]
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_acc += Custom_Metrics.voxel_accuracy(output, target)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.8f} | Val Acc: {avg_val_acc:.4%} | LR: {scheduler.get_last_lr()[0]:.4e}")

        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Val", avg_val_acc, epoch)

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
    criterion = nn.CrossEntropyLoss()

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

def job_0():
    model_constructor = partial(UNet3D_16EL, in_channels=1, out_channels=7)
    model_name = "UNet3D_SDF_16EL"
    dataset_loc = r"../../data/datasets/ABC/ABC_Data_ks_16_pad_4_bw_5_vs_adaptive_n2.torch"
    model_weights_loc = "../../data/model_weights/{model_name}/{run_name}_save_{epoch}.pth"

    epochs = 500
    backup_epochs = 100
    learning_rates = [1e-5]
    decay_order = [1e-1]
    batch_sizes = [4]

    dataset = InteractiveDataset.load_dataset(dataset_loc)
    dataset.set_split(0.75)
    dataset.split_dataset()

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
    job_0()


if __name__ == "__main__":
  main()