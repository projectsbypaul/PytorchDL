from dl_torch.models.UNet3D_Segmentation import UNet_Hilbig, UNet3D_16EL

def get_model_by_name(model_name: str, n_classes):

    model_types = {
        "UNet_Hilbig" : 0,
        "UNet_16EL" : 1
    }

    code = model_types.get(model_name)
    if code is None:
        raise ValueError(f"Unknown template name '{model_types}'")

    match code:
        case 0: return UNet_Hilbig(in_channels=1, out_channels=n_classes)
        case 1: return UNet3D_16EL(in_channels=1, out_channels=n_classes)
        case _: raise RuntimeError(f"Unhandled model type code: {code}")