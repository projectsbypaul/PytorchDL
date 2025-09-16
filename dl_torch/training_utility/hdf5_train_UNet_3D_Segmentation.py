import os
import time
import platform
import random
from platform import architecture

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from dl_torch.model_utility import Custom_Metrics
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL
from dl_torch.models.UNet3D_Segmentation import UNet_Hilbig
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


def normalize_batch_for_ce(data, target, n_classes):
    """
    Ensures:
      data   -> float32 [N, 1, D, H, W]
      target -> int64   [N, D, H, W]  (class indices)
    Accepts targets that are:
      - indices [N, D, H, W]
      - one-hot channel-first  [N, C, D, H, W]
      - one-hot channel-last   [N, D, H, W, C]
    """
    # Data channel
    if data.ndim == 4:            # [N, D, H, W]
        data = data.unsqueeze(1)  # -> [N, 1, D, H, W]
    elif data.ndim != 5:
        raise RuntimeError(f"Unexpected data shape: {tuple(data.shape)}")
    data = data.float()

    # Targets
    if target.ndim == 4:
        # already indices
        pass
    elif target.ndim == 5:
        if target.shape[1] == n_classes:          # [N, C, D, H, W]
            target = target.argmax(dim=1)
        elif target.shape[-1] == n_classes:       # [N, D, H, W, C]
            target = target.argmax(dim=-1)
        else:
            raise RuntimeError(f"Can't infer one-hot layout from target shape {tuple(target.shape)} "
                               f"with n_classes={n_classes}")
    else:
        raise RuntimeError(f"Unexpected target shape: {tuple(target.shape)}")

    return data, target.long()


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
# Helpers
# ---------------------------
def make_run_name(model_name: str, scheduler, batch_size: int) -> str:
    # Assumes scheduler exposes get_last_lr() and has .final_lr (as in your get_linear_scheduler)
    return (
        f"{model_name}"
        f"_lr[{scheduler.get_last_lr()[0]}]"
        f"_lrdc[{scheduler.final_lr / scheduler.get_last_lr()[0]:.0e}]"
        f"_bs{batch_size}"
    )


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
# Unified Training (AMP on/off)
# ---------------------------
def train_model_hdf_unified(
    model,
    dataset,
    optimizer,
    scheduler,
    criterion,
    model_name: str,
    device: torch.device,
    n_classes: int,
    *,
    use_amp: bool = False,
    backup_epoch: int = 100,
    num_epochs: int = 200,
    model_weights_loc: str | None = None,
    split: float = 0.9,
    batch_size: int = 16,
    val_batch_factor: int = 1,
    workers: int = 1,
    show_tqdm: bool = False,
    seed: int | None = None,
    resume_epoch: int | None = None,
):
    train_loader, val_loader, train_size, val_size = make_loaders(
        dataset, split, batch_size, val_batch_factor, workers, seed
    )

    print(f"Train size: {train_size}, Val size: {val_size}")
    model.to(device)

    # AMP scaler is active only if CUDA + use_amp
    scaler = GradScaler(enabled=(device.type == "cuda" and use_amp))

    run_name = make_run_name(model_name, scheduler, batch_size)
    print(f"Model: {run_name}")

    log_root = f"/logs/tensorboard/runs/{run_name}"
    writer = SummaryWriter(log_root)

    # -------- Resume logic --------
    start_epoch_idx = 0  # 0-based loop index
    if resume_epoch is not None and model_weights_loc:
        weights_path = model_weights_loc.format(
            model_name=model_name, run_name=run_name, epoch=resume_epoch
        )
        ckpt_path = weights_path.replace(".pth", ".ckpt")

        if os.path.exists(ckpt_path):
            print(f"Resuming from FULL checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if use_amp and "scaler_state_dict" in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch_idx = int(checkpoint.get("epoch_completed", resume_epoch))
        elif os.path.exists(weights_path):
            print(f"Resuming from WEIGHTS ONLY: {weights_path}")
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state)
            # Best-effort align LR schedule
            for _ in range(max(0, resume_epoch - 1)):
                scheduler.step()
            start_epoch_idx = resume_epoch
        else:
            print(
                f"WARNING: No checkpoint or weights found for epoch={resume_epoch} at\n"
                f"  {ckpt_path}\n  {weights_path}\nStarting from scratch."
            )
    # --------------------------------

    for epoch in range(start_epoch_idx, num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_train_loss, epoch_train_acc = 0.0, 0.0

        msg = f"\n[Epoch {epoch + 1}/{num_epochs}] Training..."
        tqdm.write(msg) if show_tqdm else print(msg)

        for data, target in tqdm(train_loader, desc="Training", leave=True, disable=not show_tqdm):
            data, target = normalize_batch_for_ce(data, target, n_classes)
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp and device.type == "cuda":
                with autocast(device_type="cuda", enabled=True):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                data, target = normalize_batch_for_ce(data, target, n_classes)
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                if use_amp and device.type == "cuda":
                    with autocast(device_type="cuda", enabled=True):
                        output = model(data)
                        loss = criterion(output, target)
                else:
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

        scheduler.step()
        print(f"Epoch duration: {time.time() - epoch_start:.2f} seconds")

        if ((epoch + 1) % backup_epoch) == 0 and model_weights_loc:
            # 1) Legacy weights
            backup_name = model_weights_loc.format(
                model_name=model_name, run_name=run_name, epoch=epoch + 1
            )
            os.makedirs(os.path.dirname(backup_name), exist_ok=True)
            torch.save(model.state_dict(), backup_name)

            # 2) Full checkpoint (conditionally include scaler state)
            ckpt_path = backup_name.replace(".pth", ".ckpt")
            ckpt = {
                "epoch_completed": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "run_name": run_name,
                "model_name": model_name,
                "batch_size": batch_size,
                "n_classes": n_classes,
                "use_amp": use_amp,
            }
            if use_amp and scaler is not None:
                ckpt["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt, ckpt_path)

    writer.close()
    print("Finished Training")
    return model, run_name, scaler if use_amp else None


# ---------------------------
# Orchestration (as requested name)
# ---------------------------
def train_model_hdf5(
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
    model_type: str = "default",
    resume_epoch: int | None = None,
):
    # -----------------------------------------------------------------------------
    # WARNING: Determinism in this training pipeline
    #
    # - This setup seeds RNGs and can fix split/shuffle order. It does NOT guarantee
    #   bitwise determinism due to CUDA kernels, AMP, drivers, etc.
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
    model_architecture = {
        "default": 0,
        "UNet_16EL": 1,
        "UNet_Hilbig": 2,
    }

    match model_architecture[model_type]:
        case 0:
            model = UNet3D_16EL(in_channels=1, out_channels=n_classes)
        case 1:
            model = UNet3D_16EL(in_channels=1, out_channels=n_classes)
        case 2:
            model = UNet_Hilbig(in_channels=1, out_channels=n_classes)

    # Build dataset once
    dataset = HDF5Dataset(hdf5_path)

    # One-batch sanity test (workers=0 to avoid file-handle shenanigans)
    test_loader = DataLoader(dataset, batch_size=min(2, len(dataset)),
                             shuffle=False, num_workers=0, pin_memory=False)
    data, target = next(iter(test_loader))
    data, target = normalize_batch_for_ce(data, target, n_classes)
    print("data:", tuple(data.shape), data.dtype)      # -> [N,1,D,H,W], float32
    print("target:", tuple(target.shape), target.dtype)  # -> [N,D,H,W], int64
    del test_loader  # free before creating the real loaders

    # Forward-shape sanity check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    with torch.no_grad():
        tiny_out = model(data.to(device)[:1])  # [1, C, D, H, W]
    assert tiny_out.shape[1] == n_classes, \
        f"Model out_channels={tiny_out.shape[1]} != n_classes={n_classes}"
    model.train()

    # Device & training components
    log_cuda_status()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = get_linear_scheduler(optimizer, lr, lr * decay_order, epochs)

    print(f"Using AMP: {use_amp}")

    model, run_name, scaler = train_model_hdf_unified(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        model_name=model_name,
        device=device,
        n_classes=n_classes,
        use_amp=use_amp,                 # single flag controls AMP
        backup_epoch=backup_epochs,
        num_epochs=epochs,
        model_weights_loc=model_weights_loc,
        split=split,
        batch_size=batch_size,
        val_batch_factor=val_batch_factor,
        workers=workers,
        show_tqdm=show_tqdm,
        seed=seed_for_loaders,
        resume_epoch=resume_epoch,
    )

    # Final save (weights)
    run_name_last = f"{model_name}_lr[{lr}]_lrdc[{decay_order}]bs{batch_size}"
    model_save_name = model_weights_loc.format(
        model_name=model_name, run_name=run_name_last, epoch="last"
    )
    os.makedirs(os.path.dirname(model_save_name), exist_ok=True)
    torch.save(model.state_dict(), model_save_name)

    # Final full checkpoint (optional)
    ckpt_last = {
        "epoch_completed": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "run_name": run_name_last,
        "model_name": model_name,
        "batch_size": batch_size,
        "n_classes": n_classes,
        "use_amp": use_amp,
    }
    if use_amp and scaler is not None:
        ckpt_last["scaler_state_dict"] = scaler.state_dict()
    torch.save(ckpt_last, model_save_name.replace(".pth", ".ckpt"))


def main():
    print("PyTorch:", torch.__version__)
    print("Python :", platform.python_version())

    hdf5_path = r"H:\ABC\ABC_torch\ABC_training\train_250k_ks_16_pad_4_bw_5_vs_adaptive_n3\dataset.hdf5"
    model_weights_loc = r"H:\ABC\ABC_torch\temp_models/{model_name}/{run_name}_save_{epoch}.pth"

    # Set to 0 or None for nondeterministic behavior; >0 for deterministic split/shuffle + init.
    model_seed = 1337  # try 0 to disable determinism

    model_name = "UNet_Restart"

    # Example: resume from a specific epoch if its checkpoint/weights exist
    resume_from = 3  # or None to start from scratch


    train_model_hdf5(
        model_name=model_name,
        hdf5_path=hdf5_path,
        model_weights_loc=model_weights_loc,
        epochs=5,
        backup_epochs=1,
        batch_size=8,
        lr=1e-4,
        decay_order=1e-1,
        split=0.9,
        use_amp=True,
        val_batch_factor=4,
        workers=0,
        show_tqdm=True,
        n_classes=10,
        model_seed=model_seed,
        model_type="default",
        resume_epoch=resume_from,   # resume if files exist
    )


if __name__ == "__main__":
    main()
