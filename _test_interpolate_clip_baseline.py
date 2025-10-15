# %%
import torch
from PIL import Image
import numpy as np

from app import train_mood_space, interpolate_two_images_no_compression
from my_ipadapter_model import image_grid

# %%

path1 = "/workspace/images/dog1.jpg"
path2 = "/workspace/images/fish.jpg"

def get_interpolation_images(path1, path2):
    image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
    image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")

    ws = np.linspace(0.0, 1.0, 10) # interpolation weight
    interpolated_images = interpolate_two_images_no_compression(image1, image2, ws, 
                                                n_cluster=10, match_method='hungarian', 
                                                )
    all_images = [image1] + interpolated_images + [image2]
    return all_images

# %%

all_images = get_interpolation_images(path1, path2)
img = image_grid(all_images, 2, len(all_images)//2)
display(img)

# %%



