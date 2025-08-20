import os
import time
import platform
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from dl_torch.model_utility import Custom_Metrics
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL
from dl_torch.model_utility.Scheduler import get_linear_scheduler
from dl_torch.data_utility.HDF5Dataset import HDF5Dataset


# ---------------------------
# Seeding utilities
# ---------------------------
def set_global_seeds(seed: int):
    """Seed Python, NumPy, and PyTorch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    """Windows-safe, picklable worker seeding (used only if seed>0)."""
    s = torch.initial_seed() % (2**32 - 1)
    np.random.seed(s)
    random.seed(s)


# ---------------------------
# Logging
# ---------------------------
def log_cuda_status():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number available GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"GPU index: {torch.cuda.current_device()}")
        print(torch.cuda.get_device_name(0))
        print(f"Allocated Memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Max Allocated Memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")


# ---------------------------
# DataLoader factory (handles determinism toggle)
# ---------------------------
def make_loaders(dataset,
                 split: float,
                 batch_size: int,
                 val_batch_factor: int,
                 workers: int,
                 seed: int | None):
    total_size = len(dataset)
    train_size = int(split * total_size)
    val_size = total_size - train_size

    if seed is not None and seed > 0:
        # Deterministic split/shuffle
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=gen)
        worker_fn = worker_init_fn if workers > 0 else None
    else:
        # Non-deterministic split/shuffle
        gen = None
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        worker_fn = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=gen,                     # None → nondet; set → deterministic shuffle
        num_workers=workers,
        worker_init_fn=worker_fn,
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * val_batch_factor,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=worker_fn,
        pin_memory=True,
        persistent_workers=False,
    )

    return train_loader, val_loader, train_size, val_size


# ---------------------------
# Training loops
# ---------------------------
def train_model_hdf_amp(
    model,
    dataset,
    optimizer,
    scheduler,
    criterion,
    model_name: str,
    device: torch.device,
    backup_epoch: int = 100,
    num_epochs: int = 200,
    model_weights_loc: str | None = None,
    split: float = 0.9,
    batch_size: int = 16,
    val_batch_factor: int = 1,
    workers: int = 1,
    show_tqdm: bool = False,
    seed: int | None = None,
):
    train_loader, val_loader, train_size, val_size = make_loaders(
        dataset, split, batch_size, val_batch_factor, workers, seed
    )

    print(f"Train size: {train_size}, Val size: {val_size}")
    model.to(device)

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    run_name = (
        f"{model_name}"
        f"_lr[{scheduler.get_last_lr()[0]}]"
        f"_lrdc[{scheduler.final_lr / scheduler.get_last_lr()[0]:.0e}]"
        f"_bs{batch_size}"
    )
    print(f"Model: {run_name}")

    log_root = f"/logs/tensorboard/runs/{run_name}"
    writer = SummaryWriter(log_root)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_train_loss, epoch_train_acc = 0.0, 0.0

        msg = f"\n[Epoch {epoch + 1}/{num_epochs}] Training..."
        tqdm.write(msg) if show_tqdm else print(msg)

        for data, target in tqdm(train_loader, desc="Training", leave=True, disable=not show_tqdm):
            data = data.to(device, non_blocking=True)
            target = torch.argmax(target, dim=1).long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += float(loss.item())
            epoch_train_acc += Custom_Metrics.voxel_accuracy(output, target)

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_acc = 0.0, 0.0
        msg = f"\n[Epoch {epoch + 1}/{num_epochs}] Validating..."
        tqdm.write(msg) if show_tqdm else print(msg)

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=True, disable=not show_tqdm):
                data = data.to(device, non_blocking=True)
                target = torch.argmax(target, dim=1).long().to(device, non_blocking=True)
                with autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    output = model(data)
                    loss = criterion(output, target)
                epoch_val_loss += float(loss.item())
                epoch_val_acc += Custom_Metrics.voxel_accuracy(output, target)

        # Epoch summary
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_acc / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = epoch_val_acc / len(val_loader)

        print(f"\n[Epoch {epoch+1}/{num_epochs}] Summary:")
        print(f"Train Loss: {avg_train_loss:.6f} | Train Acc: {avg_train_acc:.2%}")
        print(f"Val   Loss: {avg_val_loss:.6f} | Val   Acc: {avg_val_acc:.2%}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.4e}\n")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Val", avg_val_acc, epoch)

        torch.cuda.empty_cache()

        if ((epoch + 1) % backup_epoch) == 0 and model_weights_loc:
            backup_name = model_weights_loc.format(
                model_name=model_name, run_name=run_name, epoch=epoch + 1
            )
            os.makedirs(os.path.dirname(backup_name), exist_ok=True)
            torch.save(model.state_dict(), backup_name)

        scheduler.step()
        print(f"Epoch duration: {time.time() - epoch_start:.2f} seconds")

    writer.close()
    print("Finished Training")
    return model


def train_model_hdf(
    model,
    dataset,
    optimizer,
    scheduler,
    criterion,
    model_name: str,
    device: torch.device,
    backup_epoch: int = 100,
    num_epochs: int = 200,
    model_weights_loc: str | None = None,
    split: float = 0.9,
    batch_size: int = 16,
    val_batch_factor: int = 1,
    workers: int = 1,
    show_tqdm: bool = False,
    seed: int | None = None,
):
    train_loader, val_loader, train_size, val_size = make_loaders(
        dataset, split, batch_size, val_batch_factor, workers, seed
    )

    print(f"Train size: {train_size}, Val size: {val_size}")
    model.to(device)

    run_name = (
        f"{model_name}"
        f"_lr[{scheduler.get_last_lr()[0]}]"
        f"_lrdc[{scheduler.final_lr / scheduler.get_last_lr()[0]:.0e}]"
        f"_bs{batch_size}"
    )
    print(f"Model: {run_name}")

    log_root = f"/logs/tensorboard/runs/{run_name}"
    writer = SummaryWriter(log_root)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_train_loss, epoch_train_acc = 0.0, 0.0

        msg = f"\n[Epoch {epoch + 1}/{num_epochs}] Training..."
        tqdm.write(msg) if show_tqdm else print(msg)

        for data, target in tqdm(train_loader, desc="Training", leave=True, disable=not show_tqdm):
            data = data.to(device, non_blocking=True)
            target = torch.argmax(target, dim=1).long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_train_loss += float(loss.item())
            epoch_train_acc += Custom_Metrics.voxel_accuracy(output, target)

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_acc = 0.0, 0.0
        msg = f"\n[Epoch {epoch + 1}/{num_epochs}] Validating..."
        tqdm.write(msg) if show_tqdm else print(msg)

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=True, disable=not show_tqdm):
                data = data.to(device, non_blocking=True)
                target = torch.argmax(target, dim=1).long().to(device, non_blocking=True)
                output = model(data)
                loss = criterion(output, target)
                epoch_val_loss += float(loss.item())
                epoch_val_acc += Custom_Metrics.voxel_accuracy(output, target)

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_acc = epoch_train_acc / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_acc = epoch_val_acc / len(val_loader)

        print(f"\n[Epoch {epoch + 1}/{num_epochs}] Summary:")
        print(f"Train Loss: {avg_train_loss:.6f} | Train Acc: {avg_train_acc:.2%}")
        print(f"Val   Loss: {avg_val_loss:.6f} | Val   Acc: {avg_val_acc:.2%}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.4e}")

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", avg_train_acc, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Val", avg_val_acc, epoch)

        torch.cuda.empty_cache()

        if ((epoch + 1) % backup_epoch) == 0 and model_weights_loc:
            backup_name = model_weights_loc.format(
                model_name=model_name, run_name=run_name, epoch=epoch + 1
            )
            os.makedirs(os.path.dirname(backup_name), exist_ok=True)
            torch.save(model.state_dict(), backup_name)

        scheduler.step()
        print(f"Epoch duration: {time.time() - epoch_start:.2f} seconds")

    writer.close()
    print("Finished Training")
    return model


# ---------------------------
# Orchestration
# ---------------------------
def training_routine_hdf5(
    model_name: str,
    hdf5_path: str,
    model_weights_loc: str,
    epochs: int,
    backup_epochs: int,
    batch_size: int,
    lr,
    decay_order,
    split,
    use_amp: bool = False,
    val_batch_factor: int = 1,
    workers: int = 1,
    show_tqdm: bool = False,
    n_classes: int = 10,
    model_seed: int | None = None,
):
    # -----------------------------------------------------------------------------
    #WARNING: Determinism in this training pipeline
    #
    # - The current setup (seeded RNGs, deterministic split + shuffling, seeded
    #   workers) guarantees reproducible **data order** and **model initialization**.
    #
    # - However, this does NOT guarantee true **bitwise determinism** of training:
    #     • CUDA kernels like max-pooling and some loss ops may still be
    #       nondeterministic, depending on the backend.
    #     • Using multiple DataLoader workers (>0) introduces non-deterministic
    #       interleaving, even if worker_init_fn is seeded.
    #     • AMP (mixed precision) is inherently nondeterministic due to reduced
    #       precision and scaling heuristics.
    #     • Different GPUs, PyTorch/CUDA/cuDNN versions, or even drivers can change
    #       numerical results.
    #
    # - For strict bitwise reproducibility you would need:
    #     • torch.use_deterministic_algorithms(True)
    #     • num_workers=0
    #     • use_amp=False
    #     • deterministic-safe ops only (no max-pooling with indices, etc.)
    #
    # In practice: the current implementation should yield reproducible *statistical*
    # results (same splits, same shuffles, same seeds), but may not produce identical
    # weights or losses across runs.
    # -----------------------------------------------------------------------------

    # Conditional determinism
    if model_seed is not None and model_seed > 0:
        print(f"Deterministic mode: seeding with {model_seed}")
        set_global_seeds(model_seed)
        seed_for_loaders = model_seed
    else:
        print("Nondeterministic mode: no seeding of split/shuffle.")
        seed_for_loaders = None

    # Model & data
    model = UNet3D_16EL(in_channels=1, out_channels=n_classes)
    dataset = HDF5Dataset(hdf5_path)

    # Device & training components
    log_cuda_status()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_linear_scheduler(optimizer, lr, lr * decay_order, epochs)

    print(f"Using AMP: {use_amp}")

    train_fn = train_model_hdf_amp if use_amp else train_model_hdf

    model = train_fn(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        model_name=model_name,
        device=device,
        backup_epoch=backup_epochs,
        num_epochs=epochs,
        model_weights_loc=model_weights_loc,
        split=split,
        batch_size=batch_size,
        val_batch_factor=val_batch_factor,
        workers=workers,
        show_tqdm=show_tqdm,
        seed=seed_for_loaders,   # <- None for nondet; int for deterministic split/shuffle
    )

    # Final save
    run_name = f"{model_name}_lr[{lr}]_lrdc[{decay_order}]bs{batch_size}"
    model_save_name = model_weights_loc.format(
        model_name=model_name, run_name=run_name, epoch="last"
    )
    os.makedirs(os.path.dirname(model_save_name), exist_ok=True)
    torch.save(model.state_dict(), model_save_name)


def main():
    print("PyTorch:", torch.__version__)
    print("Python :", platform.python_version())

    hdf5_path = r"H:\ABC\ABC_torch\ABC_training\train_500k_ks_16_pad_4_bw_5_vs_adaptive_n3\train_500k_ks_16_pad_4_bw_5_vs_adaptive_n3.hdf5"
    model_weights_loc = "../../data/model_weights/{model_name}/{run_name}_save_{epoch}.pth"

    # Set to 0 or None for nondeterministic behavior; >0 for deterministic split/shuffle + init.
    model_seed = 1337  # try 0 to disable determinism

    model_name = "UNet3D_SDF_HDF5_workers_10_500k_ConditionalSeed"

    for bs in [16, 32, 64, 128]:
        training_routine_hdf5(
            model_name=model_name,
            hdf5_path=hdf5_path,
            model_weights_loc=model_weights_loc,
            epochs=2,
            backup_epochs=2,
            batch_size=bs,
            lr=1e-4,
            decay_order=1e-1,
            split=0.9,
            use_amp=True,
            val_batch_factor=4,
            workers=10,
            show_tqdm=True,
            n_classes=10,
            model_seed=model_seed,
        )


if __name__ == "__main__":
    main()
