# %%
import torch
from PIL import Image
import numpy as np

from app import train_mood_space, interpolate_two_images
from my_ipadapter_model import image_grid

# %%
path1 = "/workspace/images/dog1.jpg"
path2 = "/workspace/images/fish.jpg"

def get_interpolation_images(path1, path2, config_path):
    image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
    image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")


    model, trainer = train_mood_space([image1, image2], lr=0.001, steps=1000, config_path=config_path)

    ws = np.linspace(0.0, 1.0, 10) # interpolation weight
    interpolated_images = interpolate_two_images(image1, image2, model, ws, 
                                                n_cluster=10, match_method='hungarian', 
                                                config_path=config_path)
    all_images = [image1] + interpolated_images + [image2]
    return all_images

# %%
config_paths = ["./config_lossablate_eigvec.yaml", "./config_lossablate_flag.yaml", "./config_lossablate_recon.yaml", "./config_lossablate_recon_repluse.yaml"]

for config_path in config_paths:
    all_images = get_interpolation_images(path1, path2, config_path)
    img = image_grid(all_images, 2, len(all_images)//2)
    print()
    print("="*42)
    display(img)
    print(config_path)
    print("="*42)
    print()

# %%



