import torch
from dl_torch.models.UNet3D_Segmentation import UNet3D_16EL, UNet_Hilbig
from utility.data_exchange import cppIOexcavator
import numpy as np
import os

def __predict_in_batches(model, data, batch_size, device, use_amp=True, dtype=torch.float16, apply_softmax = False):
    model.eval()
    preds = []
    with torch.inference_mode():
        for s in range(0, len(data), batch_size):
            e = s + batch_size
            batch = data[s:e].to(device, non_blocking=True)

            if use_amp:
                # bfloat16 is safer on Ampere+; switch dtype to torch.bfloat16 if supported
                autocast_dtype = dtype if torch.cuda.is_available() else None
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    out = model(batch)
            else:
                out = model(batch)

            # argmax across class dim=1 (typical for segmentation/classification)
            if not apply_softmax:
                pred = out.argmax(dim=1).cpu().numpy()
                preds.append(pred)
            else:
                pred = out.max(dim=1).values.cpu().numpy()
                preds.append(pred)

            # free up GPU ASAP
            del out, batch
            torch.cuda.empty_cache()  # usually not needed, but helps after large peaks

    return np.concatenate(preds, axis=0)

def __run_prediction_on_dir(data_loc: str, weights_loc: str, n_classes: int, model_type:str = "UNet_Hilbig", apply_softmax: bool = False):

    # -----------------------------
    # 1) Load segments + metadata
    # -----------------------------
    data_arrays = cppIOexcavator.load_segments_from_binary(
        os.path.join(data_loc, "segmentation_data_segments.bin")
    )

    predictions_bin_pth = os.path.join(data_loc, "segmentation_data_predictions.bin")

    # Model input: (N,1,ks,ks,ks)
    model_input = torch.tensor(np.array(data_arrays)).unsqueeze(1)

    # -----------------------------
    # 2) Load model + predict
    # -----------------------------
    print("Evaluating Model")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)


    if model_type == "UNet_Hilbig":
        model = UNet_Hilbig(in_channels=1, out_channels=n_classes, apply_softmax=apply_softmax)
    elif model_type == "UNet_16EL":
        model = UNet3D_16EL(in_channels=1, out_channels=n_classes, apply_softmax=apply_softmax)
    else:
        raise NotImplementedError(f"Model type '{model_type}' not implemented")

    state_dict = torch.load(weights_loc, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("model predicting outputs...")
    device = torch.device("cuda", 0)
    batch_size = 4  # start tiny, then increase if it fits
    prediction = __predict_in_batches(model, model_input, batch_size, device, use_amp=True, dtype=torch.bfloat16, apply_softmax=apply_softmax)

    # -----------------------------
    # 3) Save predictions
    # -----------------------------
    #print(f"Saving model predictions to {predictions_bin_pth}")
    #cppIOexcavator.save_predictions(predictions_bin_pth, prediction)

    return prediction

def __assemble_grids_by_origin(grids, origins, kernel_size: int, padding: int, fill_idx):

    # assemble outputs
    origins_array = np.asarray(origins, dtype=int)

    bottom_coord   = np.min(origins_array, axis=0)
    top_inclusive  = np.max(origins_array + int(kernel_size) - 1, axis=0)
    top_exclusive  = top_inclusive + 1
    dim_vec        = (top_exclusive - bottom_coord).astype(int)

    # Offsets per block (global placement of local [0..ks) coords)
    offsets = [(-bottom_coord + np.array(o, dtype=int)) for o in origins]

    full_grid = np.full(tuple(dim_vec), fill_value=fill_idx, dtype=np.uint8)

    ks       = int(kernel_size)
    pad_half = int(padding // 2)
    x0, x1   = pad_half, ks - pad_half
    if x1 <= x0:
        print("Padding too large for kernel_size; nothing to paste.")
        return

    # Paste each predicted block (vectorized slice; no inner xyz loops)
    for g_index, off in enumerate(offsets):
        block = grids[g_index]                  # (ks, ks, ks), int64
        crop  = block[x0:x1, x0:x1, x0:x1]          # remove padding border
        dx, dy, dz = crop.shape
        ox, oy, oz = (int(off[0]) + x0, int(off[1]) + x0, int(off[2]) + x0)
        full_grid[ox:ox+dx, oy:oy+dy, oz:oz+dz] = crop.astype(np.uint8, copy=False)

    return full_grid

def main():
    pass

if __name__=="__main__":
    main()