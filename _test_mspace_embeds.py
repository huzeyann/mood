# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
config_path = "./config.yaml"
model, trainer = train_mood_space(
    pil_images=[image1, image2, image3], 
    steps=1000,
    dim=2,
    config_path=config_path,
)
# %%
from my_dino_correspondence import get_correspondence_plot, ncut_tsne_multiple_images, kway_cluster_per_image, get_single_multi_discrete_rgbs, match_centers_three_images, match_centers_two_images, get_center_features, kway_cluster_multiple_images
from my_ipadapter_model import load_ipadapter, generate
from compression_model_mkii import free_memory
from dino_clip_featextract import extract_dino_image_embeds, extract_clip_image_embeds, dino_img_transform, clip_img_transform

image_list = [image1, image2, image3]
images = torch.stack([dino_img_transform(image) for image in image_list])
dino_image_embeds = extract_dino_image_embeds(images)
images = torch.stack([clip_img_transform(image) for image in image_list])
clip_image_embeds = extract_clip_image_embeds(images)
print(dino_image_embeds.shape, clip_image_embeds.shape)
compressed_dino_image_embeds = model.compress(dino_image_embeds)
print(compressed_dino_image_embeds.shape)
# %%
n_cluster = 10
kway_eigvecs = kway_cluster_per_image(dino_image_embeds, n_cluster=n_cluster)
print(kway_eigvecs.shape)
# %%
cluster_labels = []
for i in range(images.shape[0]):
    match_pairs = match_centers_two_images(dino_image_embeds[0], dino_image_embeds[i], kway_eigvecs[0], kway_eigvecs[i], match_method='hungarian')
    _labels = kway_eigvecs[i].argmax(-1)
    matched_labels = torch.zeros_like(_labels)
    print(match_pairs)
    for i_0, i_i in enumerate(match_pairs):
        mask = _labels == i_0
        matched_labels[mask] = i_i
    cluster_labels.append(matched_labels)
        
cluster_labels = torch.stack(cluster_labels)
hw = int(np.sqrt(cluster_labels.shape[1]))
cluster_labels = cluster_labels[:, 1:].reshape(images.shape[0], hw, hw)
cluster_labels = cluster_labels.cpu().numpy()
print(cluster_labels.shape)
# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, images.shape[0], figsize=(10, 10))
for i in range(images.shape[0]):
    axes[i].imshow(cluster_labels[i], cmap='tab10')
    axes[i].set_title(f"cluster labels for image {i}")
plt.show()
# %%
n_images = compressed_dino_image_embeds.shape[0]
x2d = compressed_dino_image_embeds[:, 1:].reshape(n_images, -1, 2).detach().cpu().numpy()
color_2d = cluster_labels.reshape(n_images, -1, 1)
markers = ['o', 'x', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'h']
for i in range(n_images):
    plt.scatter(x2d[i, :, 0], x2d[i, :, 1], c=color_2d[i, :, 0], cmap='tab10', marker=markers[i], s=10)
plt.show()
# %%
fig, axes = plt.subplots(1, images.shape[0], figsize=(10, 5))
for i in range(images.shape[0]):
    _x2d = x2d[i, :, :]
    _color_2d = color_2d[i, :, :]
    axes[i].scatter(_x2d[:, 0], _x2d[:, 1], c=_color_2d[:, 0], cmap='tab10', marker=markers[i], s=10)
    axes[i].set_title(f"mspace embeds for image {i}")
plt.show()
# %%