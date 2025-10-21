# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from PIL import Image
import numpy as np
from app import train_mood_space, analogy_three_images
from my_ipadapter_model import image_grid
import matplotlib.pyplot as plt
# 
path1 = "./images/jimi_portrait.jpg"
path2 = "./images/jimi_action.jpg"
path3 = "./images/bach_portrait.jpg"
path4 = "./images/violin.jpg"
image1 = Image.open(path1).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image2 = Image.open(path2).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image3 = Image.open(path3).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
image4 = Image.open(path4).resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
# %%
grid = image_grid([image1, image2, image3, image4], 1, 4)
grid
# %%
for _ in range(10):
    config_path = "./config.yaml"
    model, trainer = train_mood_space(
        pil_images=[image1, image2, image3, image4], 
        steps=1000,
        config_path=config_path,
    )
    interpolation_weights = np.linspace(0.0, 2.0, 10).tolist()
    correspondence_plot, fig, interpolated_images = analogy_three_images(
        [image3, image1, image2], 
        model, 
        interpolation_weights,
        n_cluster=20,
        match_method='hungarian',
    )
    all_images = interpolated_images

    display_size = (256, 256)
    resized_images = [img.resize(display_size, Image.Resampling.LANCZOS) for img in all_images]
    result_grid = image_grid(resized_images, 2, len(resized_images)//2)
    plt.show()
    display(result_grid)


# %%