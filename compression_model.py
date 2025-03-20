from collections import defaultdict
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ncut_pytorch import nystrom_ncut
from ncut_pytorch.ncut_pytorch import find_gamma_by_degree_after_fps
from ncut_pytorch import NCUT, kway_ncut
from ncut_pytorch.ncut_pytorch import find_gamma_by_degree_after_fps

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from riemann_curvature_loss import compute_riemann_curvature_loss, compute_boundary_loss, compute_repulsion_loss
from riemann_curvature_loss import compute_axis_align_loss

import gradio as gr

def nystrom_ncut_wrapper(features, n_eig, degree=None, gamma=0.5, distance='rbf', max_num_sample=2056):
    num_sample = min(max_num_sample, features.shape[0]//4)
    if gamma is None:
        with torch.no_grad():
            gamma = find_gamma_by_degree_after_fps(features, degree=degree, num_sample=num_sample, distance=distance, max_iter=10)
            
    n_eig += 6  # for better svd_lowrank
    n_eig = min(n_eig, features.shape[0]//4)
    eigvec, eigval, sampled_indices = nystrom_ncut(features, n_eig, 
                                num_sample=num_sample, sample_method='farthest',
                                distance=distance, affinity_focal_gamma=gamma,
                                indirect_connection=False, make_orthogonal=False)
    eigvec = eigvec[:, :-6]
    eigval = eigval[:-6]
    return eigvec, eigval


def nystrom_ncut_wrapper_safe(*args, **kwargs):
    features = args[0]
    n_eig = args[1]
    if torch.any(features.isnan()):
        raise ValueError("input contains NaN values")
    
    try:
        return nystrom_ncut_wrapper(*args, **kwargs)
    except:
        logging.warning("nystrom_ncut_wrapper failed, returning zeros")
        eigvec = torch.zeros((features.shape[0], n_eig), device=features.device)
        eigval = torch.zeros((n_eig,), device=features.device)
        return eigvec, eigval

def _kway_ncut_loss(eigvec_gt, eigvec_hat, n_eig):
    _eigvec_gt = eigvec_gt[:, :n_eig]
    _eigvec_hat = eigvec_hat[:, :n_eig]
    loss = F.smooth_l1_loss(_eigvec_gt @ _eigvec_gt.T, _eigvec_hat @ _eigvec_hat.T)
    return loss

def hierarchical_kway_ncut_loss(eigvec_gt, eigvec_hat, n_eig, start=4, step_mult=2):
    if torch.all(eigvec_gt == 0) or torch.all(eigvec_hat == 0):
        return torch.tensor(0, device=eigvec_gt.device)
    
    loss = 0
    n_eig = start // step_mult
    while True:
        n_eig *= step_mult
        loss += _kway_ncut_loss(eigvec_gt, eigvec_hat, n_eig)
        if n_eig > eigvec_gt.shape[1] or n_eig > eigvec_hat.shape[1]:
            break
    return loss



@torch.no_grad()
def get_fg_mask(image_embeds, num_clusters=3):
    # image_embeds b, l, c
    if image_embeds.dim() == 2:
        image_embeds = image_embeds.unsqueeze(0)
    b, l, c = image_embeds.shape
    hw = int(np.sqrt(l))
    inp = image_embeds[:, 1:].reshape(b*hw*hw, c)
    gamma = find_gamma_by_degree_after_fps(inp, 0.1, distance='rbf')
    eigvec, eigval = NCUT(10, affinity_focal_gamma=gamma, distance='rbf', device='cuda').fit_transform(inp)
    kway_onehot = kway_ncut(eigvec[:, :num_clusters])
    kway_index = kway_onehot.argmax(dim=-1)
    kway_index = kway_index.reshape(b, hw, hw)
    centers = kway_index[:, 8, 8]
    corners = torch.cat([kway_index[:, 0, 0], kway_index[:, 0, 15], kway_index[:, 15, 0], kway_index[:, 15, 15]], dim=0)
    
    center_mode = centers.mode().values.item()
    corner_mode = corners.mode().values.item()
    
    fg_mask = kway_index == center_mode
    fg_mask = fg_mask.reshape(b, hw*hw)
    # add back the first token
    fg_mask = torch.cat([torch.ones((b, 1), device=fg_mask.device), fg_mask], dim=1)
    fg_mask = fg_mask.bool()
    return fg_mask


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer=4, latent_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.GELU(),
            *[nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU()) for _ in range(n_layer)],
            nn.Linear(latent_dim, out_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


class CompressionModel(pl.LightningModule):
    def __init__(self, cfg, gradio_progress=False):
        super().__init__()
        
        self.compress = MLP(cfg.in_dim, cfg.mood_dim, cfg.n_layer, cfg.latent_dim)
        self.uncompress = MLP(cfg.mood_dim, cfg.out_dim, cfg.n_layer, cfg.latent_dim)
        self.uncompress_dummy = MLP(cfg.mood_dim, cfg.in_dim, cfg.n_layer, cfg.latent_dim)
                
        self.cfg = cfg

        self.loss_history = defaultdict(list)
        self.gradio_progress = gradio_progress
        self.progress = gr.Progress()

    def training_step(self, batch):
        if self.gradio_progress and self.trainer.global_step % 10 == 0 and self.trainer.global_step > 0:
            self.progress(self.trainer.global_step/self.cfg.steps, desc=f"Training, loss = {self.loss_history['recon'][-1]:.4f}")

        feats = batch[0]
        target_feats = batch[1]
        fg_masks = batch[2].flatten()
        feats_compressed = self.compress(feats)
        feats_uncompressed = self.uncompress(feats_compressed)
        feats_uncompressed_dummy = self.uncompress_dummy(feats_compressed)
        
        eigvec_gt, eigval_gt = nystrom_ncut_wrapper_safe(feats[fg_masks], self.cfg.n_eig)
        eigvec_hat, eigval_hat = nystrom_ncut_wrapper_safe(feats_compressed, self.cfg.n_eig)
        eigvec_hat = eigvec_hat[fg_masks]

        total_loss = 0
        if self.cfg.eigvec_loss > 0:
            eigvec_loss = hierarchical_kway_ncut_loss(eigvec_gt, eigvec_hat, n_eig=self.cfg.n_eig)
            self.log("loss/eigvec", eigvec_loss, prog_bar=True)
            total_loss += eigvec_loss * self.cfg.eigvec_loss
            self.loss_history['eigvec'].append(eigvec_loss.item())

        if self.cfg.recon_loss_fg > 0 and torch.any(fg_masks):
            recon_loss_fg = F.smooth_l1_loss(target_feats[fg_masks], feats_uncompressed[fg_masks])
            self.log("loss/recon_fg", recon_loss_fg, prog_bar=True)
            total_loss += recon_loss_fg * self.cfg.recon_loss_fg
            self.loss_history['recon'].append(recon_loss_fg.item())

            recon_loss_fg_dummy = F.smooth_l1_loss(feats[fg_masks], feats_uncompressed_dummy[fg_masks])
            self.log("loss/recon_fg_dummy", recon_loss_fg_dummy, prog_bar=True)
            total_loss += recon_loss_fg_dummy * self.cfg.recon_loss_fg_dummy

        if self.cfg.recon_loss_bg > 0 and not torch.all(fg_masks):
            recon_loss_bg = F.smooth_l1_loss(target_feats[~fg_masks], feats_uncompressed[~fg_masks])
            self.log("loss/recon_bg", recon_loss_bg, prog_bar=True)
            total_loss += recon_loss_bg * self.cfg.recon_loss_bg

            recon_loss_bg_dummy = F.smooth_l1_loss(feats[~fg_masks], feats_uncompressed_dummy[~fg_masks])
            self.log("loss/recon_bg_dummy", recon_loss_bg_dummy, prog_bar=True)
            total_loss += recon_loss_bg_dummy * self.cfg.recon_loss_bg_dummy

        if self.cfg.riemann_curvature_loss > 0:
            riemann_curvature_loss = compute_riemann_curvature_loss(feats_compressed[fg_masks])
            self.log("loss/riemann_curvature", riemann_curvature_loss, prog_bar=True)
            total_loss += riemann_curvature_loss * self.cfg.riemann_curvature_loss

        if self.cfg.axis_align_loss > 0:
            axis_align_loss = compute_axis_align_loss(feats_compressed[fg_masks])
            self.log("loss/axis_align", axis_align_loss, prog_bar=True)
            total_loss += axis_align_loss * self.cfg.axis_align_loss

        if self.cfg.repulsion_loss > 0:
            repulsion_loss = compute_repulsion_loss(feats_compressed[fg_masks])
            self.log("loss/repulsion", repulsion_loss, prog_bar=True)
            total_loss += repulsion_loss * self.cfg.repulsion_loss

        if self.cfg.boundary_loss > 0:
            boundary_loss = compute_boundary_loss(feats_compressed)
            self.log("loss/boundary", boundary_loss, prog_bar=True)
            total_loss += boundary_loss * self.cfg.boundary_loss

        loss = total_loss
        self.log("loss/total", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.NAdam(self.parameters(), lr=self.cfg.lr)
        return optimizer

class DatasetWithSimplices(torch.utils.data.Dataset):
    def __init__(self, input_feats, target_feats, plus_masks):
        self.input_feats = input_feats
        self.target_feats = target_feats
        self.plus_masks = plus_masks
    def __len__(self):
        return len(self.input_feats)
    def __getitem__(self, idx):
        return self.input_feats[idx], self.target_feats[idx], self.plus_masks[idx]


def free_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    import gc
    gc.collect()


def train_compression_model(model, cfg: DictConfig, input_feats, target_feats, 
                            plus_masks=None, devices=[0], compute_fg_mask=False):
    free_memory()
    b, l, c = input_feats.shape
    if compute_fg_mask and plus_masks is None:
        plus_masks = get_fg_mask(input_feats)
    if plus_masks is None:
        plus_masks = torch.ones((b*l)).bool()
    plus_masks = plus_masks.flatten()
    input_feats = input_feats.flatten(end_dim=-2)
    target_feats = target_feats.flatten(end_dim=-2)

    logger = pl.loggers.TensorBoardLogger(cfg.log_dir, name=cfg.name)
    trainer = pl.Trainer(max_steps=cfg.steps,
                         gradient_clip_val=cfg.grad_clip_val,
                         accelerator="gpu", 
                         devices=devices,
                         enable_checkpointing=False,
                         logger=logger,
    )
    dataset = DatasetWithSimplices(input_feats, target_feats, plus_masks)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    trainer.fit(model, dataloader)

    return trainer