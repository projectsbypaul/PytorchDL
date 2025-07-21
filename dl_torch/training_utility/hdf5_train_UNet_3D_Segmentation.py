import torch
import torch.nn as nn
from dl_torch.model_utility import Custom_Metrics
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL
import os
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from dl_torch.model_utility.Scheduler import get_linear_scheduler
from torch.utils.data import random_split, DataLoader
from dl_torch.data_utility.HDF5Dataset import HDF5Dataset
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import time
import platform

def log_cuda_status():
    print(f"CUDA available: {torch.cuda.is_available()}")  # True if a GPU is available
    print(f"Number available GPUs {torch.cuda.device_count()}")  # Number of GPUs
    print(f"GPU index {torch.cuda.current_device()}")  # Current GPU index
    print(torch.cuda.get_device_name(0))
    print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

def train_model_hdf_amp(model,
                    dataset,
                    optimizer,
                    scheduler,
                    criterion,
                    model_name: str,
                    device: torch.device,
                    backup_epoch: int = 100,
                    num_epochs=200,
                    model_weights_loc: str = None,
                    split: float = 0.9,
                    batch_size: int = 16,
                    val_batch_factor : int = 1,
                    workers: int = 1,
                    show_tqdm: bool = False):

    total_size = len(dataset)
    train_size = int(split * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * val_batch_factor, shuffle=False, num_workers=workers, pin_memory=True)

    print(f"Train size: {train_size}, Val size: {val_size}")
    model.to(device)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    run_name = (f"{model_name}"
                f"_lr[{scheduler.get_last_lr()[0]}]"
                f"_lrdc[{scheduler.final_lr / scheduler.get_last_lr()[0]:.0e}]"
                f"_bs{batch_size}")
    print(f"Model: {run_name}")

    log_root = f"logs/tensorboard/runs/{run_name}"
    writer = SummaryWriter(log_root)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_train_loss, epoch_train_acc = 0.0, 0.0

        msg = f"\n[Epoch {epoch + 1}/{num_epochs}] Training..."
        if show_tqdm:
            tqdm.write(msg)
        else:
            print(msg)

        for data, target in tqdm(train_loader, desc="Training", leave=True, disable=not show_tqdm):
            data = data.to(device)
            target = torch.argmax(target, dim=1).long().to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            epoch_train_acc += Custom_Metrics.voxel_accuracy(output, target)

        model.eval()
        epoch_val_loss, epoch_val_acc = 0.0, 0.0

        msg = f"\n[Epoch {epoch + 1}/{num_epochs}] Validating..."
        if show_tqdm:
            tqdm.write(msg)
        else:
            print(msg)

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=True, disable=not show_tqdm):
                data = data.to(device)
                target = torch.argmax(target, dim=1).long().to(device)

                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    output = model(data)
                    loss = criterion(output, target)

                epoch_val_loss += loss.item()
                epoch_val_acc += Custom_Metrics.voxel_accuracy(output, target)

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_acc / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = epoch_val_acc / len(val_loader)

        print(f"\n[Epoch {epoch+1}/{num_epochs}] Summary:")
        print(f"Train Loss: {avg_train_loss:.6f} | Train Acc: {avg_train_acc:.2%}")
        print(f"Val Loss:   {avg_val_loss:.6f} | Val Acc:   {avg_val_acc:.2%}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.4e}\n")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Val", avg_val_acc, epoch)

        torch.cuda.empty_cache()

        if (epoch % backup_epoch) == 0 and model_weights_loc:
            backup_name = model_weights_loc.format(model_name=model_name, run_name=run_name, epoch=epoch)
            os.makedirs(os.path.dirname(backup_name), exist_ok=True)
            torch.save(model.state_dict(), backup_name)

        scheduler.step()
        print(f"Epoch duration: {time.time() - epoch_start:.2f} seconds")

    writer.close()
    print("Finished Training")
    return model


def train_model_hdf(model,
                    dataset,
                    optimizer,
                    scheduler,
                    criterion,
                    model_name: str,
                    device: torch.device,
                    backup_epoch: int = 100,
                    num_epochs=200,
                    model_weights_loc: str = None,
                    split: float = 0.9,
                    batch_size: int = 16,
                    val_batch_factor: int = 1,
                    workers: int = 1,
                    show_tqdm: bool = False):

    total_size = len(dataset)
    train_size = int(split * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*val_batch_factor, shuffle=False, num_workers=workers, pin_memory=True)

    print(f"Train size: {train_size}, Val size: {val_size}")
    model.to(device)

    run_name = (f"{model_name}"
                f"_lr[{scheduler.get_last_lr()[0]}]"
                f"_lrdc[{scheduler.final_lr / scheduler.get_last_lr()[0]:.0e}]"
                f"_bs{batch_size}")
    print(f"Model: {run_name}")

    log_root = f"logs/tensorboard/runs/{run_name}"
    writer = SummaryWriter(log_root)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_train_loss, epoch_train_acc = 0.0, 0.0

        msg = f"\n[Epoch {epoch + 1}/{num_epochs}] Training..."
        if show_tqdm:
            tqdm.write(msg)
        else:
            print(msg)

        for data, target in tqdm(train_loader, desc="Training", leave=True, disable=not show_tqdm):
            data = data.to(device)
            target = torch.argmax(target, dim=1).long().to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_acc += Custom_Metrics.voxel_accuracy(output, target)

        model.eval()
        epoch_val_loss, epoch_val_acc = 0.0, 0.0

        msg = f"\n[Epoch {epoch + 1}/{num_epochs}] Validating..."
        if show_tqdm:
            tqdm.write(msg)
        else:
            print(msg)

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=True, disable=not show_tqdm):
                data = data.to(device)
                target = torch.argmax(target, dim=1).long().to(device)
                output = model(data)
                loss = criterion(output, target)
                epoch_val_loss += loss.item()
                epoch_val_acc += Custom_Metrics.voxel_accuracy(output, target)

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_acc / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = epoch_val_acc / len(val_loader)

        print(f"\n[Epoch {epoch + 1}/{num_epochs}] Summary:")
        print(f"Train Loss: {avg_train_loss:.6f} | Train Acc: {avg_train_acc:.2%}")
        print(f"Val Loss:   {avg_val_loss:.6f} | Val Acc:   {avg_val_acc:.2%}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.4e}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Val", avg_val_acc, epoch)

        torch.cuda.empty_cache()

        if (epoch % backup_epoch) == 0 and model_weights_loc:
            backup_name = model_weights_loc.format(model_name=model_name, run_name=run_name, epoch=epoch + 1)
            os.makedirs(os.path.dirname(backup_name), exist_ok=True)
            torch.save(model.state_dict(), backup_name)

        scheduler.step()
        print(f"Epoch duration: {time.time() - epoch_start:.2f} seconds")

    writer.close()
    print("Finished Training")
    return model

def training_routine_hdf5(model_name : str,
                          hdf5_path : str,
                          model_weights_loc : str,
                          epochs : int,
                          backup_epochs : int,
                          batch_size : int,
                          lr,
                          decay_order,
                          split,
                          use_amp: bool =False,
                          val_batch_factor: int = 1,
                          workers: int = 1,
                          show_tqdm: bool = False):
    # Model setup
    model = UNet3D_16EL(in_channels=1, out_channels=10)
    dataset = HDF5Dataset(hdf5_path)

    log_cuda_status()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_linear_scheduler(optimizer, lr, lr * decay_order, epochs)

    print(f"Using AMP: {use_amp}")

    train_model = train_model_hdf_amp if use_amp else train_model_hdf

    model = train_model(model,
                     dataset,
                     optimizer,
                     scheduler,
                     criterion,
                     model_name,
                     device,
                     backup_epochs,
                     epochs,
                     model_weights_loc,
                     split,
                     batch_size,
                     val_batch_factor,
                     workers,
                     show_tqdm)

    run_name = f"{model_name}_lr[{lr}]_lrdc[{decay_order}]bs{batch_size}"
    model_save_name = model_weights_loc.format(model_name=model_name, run_name=run_name, epoch="last")
    torch.save(model.state_dict(), model_save_name)

def main():
    print(torch.__version__)

    hdf5_path = r"H:\ABC\ABC_torch\ABC_training\train_250k_ks_16_pad_4_bw_5_vs_adaptive_n3\dataset.hdf5"
    model_weights_loc = "../../data/model_weights/{model_name}/{run_name}_save_{epoch}.pth"

    model_name = "UNet3D_SDF_HDF5_workers_14_250k_AMP"
    training_routine_hdf5(model_name,
                          hdf5_path,
                          model_weights_loc,
                          epochs=2,
                          backup_epochs=2,
                          batch_size=16,
                          lr=1e-4,
                          decay_order=1e-1,
                          split=0.9,
                          use_amp=True,
                          val_batch_factor=1,
                          workers=14,
                          show_tqdm=False)  # <-- toggle here

    model_name = "UNet3D_SDF_HDF5_workers_14_250k"
    training_routine_hdf5(model_name,
                          hdf5_path,
                          model_weights_loc,
                          epochs=2,
                          backup_epochs=2,
                          batch_size=16,
                          lr=1e-4,
                          decay_order=1e-1,
                          split=0.9,
                          use_amp=False,
                          val_batch_factor=1,
                          workers=14,
                          show_tqdm=False)  # <-- toggle here

if __name__ == "__main__":
    main()
