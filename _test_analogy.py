# %%
import torch
from PIL import Image
import numpy as np

from app import train_mood_space, interpolate_two_images, analogy_three_images
from my_ipadapter_model import image_grid

# %%

config_path = "./config_lossablate_eigvec.yaml"

path1 = "/workspace/images/jimi_portrait.jpg"
path2 = "/workspace/images/jimi_action.jpg"
path3 = "/workspace/images/bach_portrait.jpg"

image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image3 = Image.open(path3).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")

model, trainer = train_mood_space([image1, image2, image3], lr=0.001, steps=1000, config_path=config_path)

ws = np.linspace(0.0, 1.0, 10) # interpolation weight
correspondence_image, fig, interpolated_images = analogy_three_images([image3, image1, image2], model, ws, 
                                            n_cluster=10, match_method='hungarian', 
                                            config_path=config_path)
all_images = [image1] + interpolated_images + [image2]

# %%

img = image_grid(all_images, 2, len(all_images)//2)
display(img)

# %%



