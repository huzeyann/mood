import numpy as np
from PIL import Image
from featextract_utils import img_transform_inv
from ipadapter import image_grid

from ncut_pytorch import NCUT, kway_ncut, rgb_from_tsne_3d, convert_to_lab_color
from ncut_pytorch.ncut_pytorch import find_gamma_by_degree
from einops import rearrange
import torch


def ncut_tsne_multiple_images(image_embeds, n_eig=50, gamma=0.5, degree=0.5):
    b, l, c = image_embeds.shape
    inp = image_embeds.flatten(end_dim=-2)
    if gamma is None:
        gamma = find_gamma_by_degree(inp, degree, distance='rbf')
    eigvec, eigval = NCUT(n_eig, affinity_focal_gamma=gamma, distance='rbf', device='cuda').fit_transform(inp)
    x3d, rgb = rgb_from_tsne_3d(eigvec, device='cuda', perplexity=50)
    rgb = convert_to_lab_color(rgb)
    rgb = rearrange(rgb, '(b l) c -> b l c', b=b)
    eigvec = rearrange(eigvec, '(b l) c -> b l c', b=b)
    return eigvec, rgb

def _kway_cluster_one_image(image_embeds, n_cluster, gamma=0.5, degree=0.5):
    l, c = image_embeds.shape
    inp = image_embeds.flatten(end_dim=-2)
    if gamma is None:
        gamma = find_gamma_by_degree(inp, degree, distance='rbf')
    n_eig = n_cluster * 2 + 6
    n_eig = min(n_eig, inp.shape[0]//2-1)
    num_samples = min(1000, inp.shape[0]//2)
    eigvec, eigval = NCUT(n_eig, num_sample=num_samples, 
                          affinity_focal_gamma=gamma, distance='rbf', device='cuda').fit_transform(inp)
    eigvec_onehot, eigvec_continues = kway_ncut(eigvec[:, :n_cluster], return_continuous=True)
    return eigvec_continues

def kway_cluster_per_image(image_embeds, n_cluster, gamma=0.5, degree=0.5):
    eigvecs = []
    for i in range(image_embeds.shape[0]):
        eigvec = _kway_cluster_one_image(image_embeds[i], n_cluster, gamma, degree)
        eigvecs.append(eigvec)
    eigvecs = torch.stack(eigvecs)
    return eigvecs

def kway_cluster_multiple_images(image_embeds, n_cluster, gamma=0.5, degree=0.5):
    b, l, c = image_embeds.shape
    inp = image_embeds.flatten(end_dim=-2)
    if gamma is None:
        gamma = find_gamma_by_degree(inp, degree, distance='rbf')
    n_eig = n_cluster * 2 + 6
    n_eig = min(n_eig, inp.shape[0]//2-1)
    num_samples = min(1000, inp.shape[0]//2)
    eigvec, eigval = NCUT(n_eig, num_sample=num_samples, 
                          affinity_focal_gamma=gamma, distance='rbf', device='cuda').fit_transform(inp)
    eigvec_onehot, eigvec_continues = kway_ncut(eigvec[:, :n_cluster], return_continuous=True)
    eigvec_continues = rearrange(eigvec_continues, '(b l) c -> b l c', b=b)
    return eigvec_continues


def get_single_multi_discrete_rgbs(joint_rgbs, single_eigvecs):
    n_cluster = single_eigvecs.shape[-1]
    discrete_rgbs = np.zeros_like(joint_rgbs)
    for i_img in range(joint_rgbs.shape[0]):
        _rgb = joint_rgbs[i_img]
        _eigvec = single_eigvecs[i_img].cpu().numpy()
        _cluster_labels = _eigvec.argmax(-1)
        _discrete_rgb = np.zeros_like(_rgb)
        for i_cluster in range(n_cluster):
            _discrete_rgb[_cluster_labels == i_cluster] = _rgb[_cluster_labels == i_cluster].mean(0)
        discrete_rgbs[i_img] = _discrete_rgb
    discrete_rgbs = discrete_rgbs * 255
    discrete_rgbs = discrete_rgbs.astype(np.uint8)
    return discrete_rgbs


def get_center_features(image_embeds, cluster_labels, n_cluster):
    center_features = torch.zeros((n_cluster, image_embeds.shape[-1]))
    for i_cluster in range(n_cluster):
        mask = cluster_labels == i_cluster
        if mask.sum() > 0:
            center_features[i_cluster] = image_embeds[mask].mean(0)
        else:
            # center_features[i_cluster] = torch.zeros_like(image_embeds[0])
            center_features[i_cluster] = torch.ones_like(image_embeds[0]) * 114514
    return center_features

def cosine_similarity(A, B):
    _A = A / A.norm(dim=-1, keepdim=True)
    _B = B / B.norm(dim=-1, keepdim=True)
    return _A @ _B.T

from scipy.optimize import linear_sum_assignment
def hungarian_match_centers(center_features1, center_features2):
    dist = torch.cdist(center_features1, center_features2)
    dist = dist.cpu().detach().numpy()
    row_ind, col_ind = linear_sum_assignment(dist)
    return col_ind

def argmin_matching(center_features1, center_features2):
    dist = torch.cdist(center_features1, center_features2)
    dist = dist.cpu().detach().numpy()
    return np.argmin(dist, axis=-1)


def match_centers(image_embed1, image_embed2, eigvec1, eigvec2, match_method='hungarian'):
    cluster_label1 = eigvec1.argmax(-1).cpu().numpy()
    cluster_label2 = eigvec2.argmax(-1).cpu().numpy()
    n_cluster = eigvec1.shape[-1]
    center_features1 = get_center_features(image_embed1, cluster_label1, n_cluster=n_cluster)
    center_features2 = get_center_features(image_embed2, cluster_label2, n_cluster=n_cluster)
    if match_method == 'hungarian':
        one_to_one_mapping = hungarian_match_centers(center_features1, center_features2)
    elif match_method == 'argmin':
        one_to_one_mapping = argmin_matching(center_features1, center_features2)
    return one_to_one_mapping



def match_centers_three_images(image_embeds, eigvecs, match_method='hungarian'):
    # image_embeds: b, l, c; b = 3, A2, A1, B1
    # eigvecs: b, l
    A2_to_A1 = match_centers(image_embeds[0], image_embeds[1], eigvecs[0], eigvecs[1], match_method=match_method)
    A1_to_B1 = match_centers(image_embeds[1], image_embeds[2], eigvecs[1], eigvecs[2], match_method=match_method)

    return A2_to_A1, A1_to_B1

def match_centers_two_images(image_embed1, image_embed2, eigvec1, eigvec2, match_method='hungarian'):
    one_to_one_mapping = match_centers(image_embed1, image_embed2, eigvec1, eigvec2, match_method=match_method)
    return one_to_one_mapping
    


def plot_clusters(image, eigvec, cluster_order, hw=16):
    cluster_images = []
    img = img_transform_inv(image).resize((128, 128), resample=Image.Resampling.NEAREST)
    for idx_cluster in cluster_order:
        mask = eigvec.argmax(-1) == idx_cluster
        mask = mask.cpu().numpy()[1:].reshape(hw, hw)
        mask = (mask * 255).astype(np.uint8)
        mask = Image.fromarray(mask).resize((128, 128), resample=Image.Resampling.NEAREST)
        # superimpose
        mask = np.array(mask).astype(np.float32) / 255
        _img = np.array(img).astype(np.float32) / 255
        mask = np.stack([mask] * 3, axis=-1)
        mask[mask == 0] = 0.1
        _img = _img * mask
        _img = _img * 255
        _img = _img.astype(np.uint8)
        cluster_images.append(Image.fromarray(_img))
    return cluster_images


def grid_one_image(image, eigvec, cluster_order, discrete_rgb, hw=16, n_cols=10):
    cluster_images = plot_clusters(image, eigvec, cluster_order, hw)
    img = img_transform_inv(image).resize((128, 128), resample=Image.Resampling.NEAREST)
    ncut_image = discrete_rgb[1:].reshape(hw, hw, 3)
    ncut_image = Image.fromarray(ncut_image).resize((128, 128), resample=Image.Resampling.NEAREST)

    # extend cluster_images to n_cols
    num_missing = n_cols - len(cluster_images) % n_cols
    num_missing = 0 if num_missing == n_cols else num_missing
    _img_append = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    cluster_images.extend([_img_append] * num_missing)

    # add img and ncut_image before each row
    prepend_images = [img, ncut_image]
    n_rows = len(cluster_images) // n_cols
    new_cluster_images = []
    for i_row in range(n_rows):
        image_list = prepend_images + cluster_images[i_row * n_cols:(i_row + 1) * n_cols]
        new_cluster_images.append(image_list)
    return new_cluster_images


def grid_multiple_images(images, eigvecs, cluster_orders, discrete_rgbs, hw=16, n_cols=10):
    grid_images = []
    for image, eigvec, cluster_order, discrete_rgb in zip(images, eigvecs, cluster_orders, discrete_rgbs):
        grid_images.append(grid_one_image(image, eigvec, cluster_order, discrete_rgb, hw, n_cols))
    
    interleave_images = []
    for i_row in range(len(grid_images[0])):
        for i_img in range(len(grid_images)):
            interleave_images.append(grid_images[i_img][i_row])
    return interleave_images


def get_correspondence_plot(images, eigvecs, cluster_orders, discrete_rgbs, hw=16, n_cols=10):
    n_cluster = eigvecs.shape[-1]
    n_cols = min(n_cols, n_cluster)
    interleave_images = grid_multiple_images(images, eigvecs, cluster_orders, discrete_rgbs, hw, n_cols)
    n_row = len(interleave_images)
    n_cols = len(interleave_images[0])
    grid = image_grid(sum(interleave_images, []), n_row, n_cols)
    return grid