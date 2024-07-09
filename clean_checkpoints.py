from pathlib import Path
import torch
from beat_this.inference import lightning_to_torch, load_model

# load checkpoint
checkpoint_id = "bfyxbq91"
out_name = "fold7.ckpt"
convert_to_pure_torch = False

# infer the path from the run id
folder = Path("JBT", checkpoint_id, "checkpoints")
potential_ckpts = list(folder.glob("*.ckpt"))
if len(potential_ckpts) != 1:
    raise ValueError(
        f"Expected one .ckpt file in {folder}, found {len(potential_ckpts)}")
checkpoint_path = potential_ckpts[0]

if not convert_to_pure_torch:
    # clean and keep only the keys "state_dict" and "datamodule"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cleaned_checkpoint = {k: v for k, v in checkpoint.items() if k in ["state_dict", "datamodule_hyper_parameters","hyper_parameters","pytorch-lightning_version"]}
    # save the cleaned checkpoint
    cleaned_checkpoint_path = Path("checkpoints",out_name)
    torch.save(cleaned_checkpoint, cleaned_checkpoint_path)
else:
    # convert to pure torch
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cleaned_checkpoint = lightning_to_torch(checkpoint)
    # load the model
    model = load_model(checkpoint_path, torch.device("cpu"))
    # save the model
    model_path = Path("checkpoints",out_name)
    torch.save(model.state_dict(), model_path)



