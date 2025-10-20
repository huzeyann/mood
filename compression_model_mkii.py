from collections import defaultdict
import logging
from einops import rearrange
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

from ncut_pytorch.ncut_pytorch import affinity_from_features, ncut
from ncut_pytorch.ncut_pytorch import nystrom_ncut
from ncut_pytorch.affinity_gamma import find_gamma_by_degree_after_fps
from ncut_pytorch.math_utils import compute_riemann_curvature_loss, compute_boundary_loss, compute_repulsion_loss, compute_axis_align_loss

def _kway_ncut_loss(eigvec_gt, eigvec_hat, n_eig):
    _eigvec_gt = eigvec_gt[:, :n_eig]
    _eigvec_hat = eigvec_hat[:, :n_eig]
    loss = F.smooth_l1_loss(_eigvec_gt @ _eigvec_gt.T, _eigvec_hat @ _eigvec_hat.T)
    return loss

def flag_space_loss(eigvec_gt, eigvec_hat, n_eig, start=4, step_mult=2):
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

def ncut_wrapper(features, n_eig, distance='rbf'):
    A = affinity_from_features(features, distance=distance)
    eigvec, eigval = ncut(A, n_eig)
    return eigvec, eigval



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
    
class PoolingCNN(nn.Module):
    def __init__(self, n_chan, downsample=4):
        super().__init__()
        self.cnn = nn.Conv2d(n_chan, n_chan, kernel_size=downsample, stride=downsample)

    def forward(self, x):
        # Accepts x with shape (b, l, c) or (l, c)
        added_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            added_batch = True
        elif x.dim() != 3:
            raise ValueError(f"Expected input of shape (b, l, c) or (l, c), got {tuple(x.shape)}")

        b, l, c = x.shape
        if l < 2:
            raise ValueError("Sequence length l must be at least 2 (1 cls token + at least 1 patch token).")

        # l is assumed to be a perfect square + 1 cls token
        hw = int(round((l - 1) ** 0.5))
        if hw * hw != (l - 1):
            raise ValueError(f"l-1 must be a perfect square. Got l={l}, l-1={l-1}.")

        cls_token = x[:, :1, :]  # b, 1, c
        feat_tokens = rearrange(x[:, 1:, :], 'b (h w) c -> b c h w', h=hw, w=hw)  # b, c, h, w
        pooled_feat_tokens = self.cnn(feat_tokens)  # b, c, h', w'
        pooled_feat_tokens = rearrange(pooled_feat_tokens, 'b c h w -> b (h w) c')  # b, h'*w', c
        out = torch.cat([cls_token, pooled_feat_tokens], dim=1)  # b, 1+h'*w', c

        if added_batch:
            out = out.squeeze(0)  # return (l, c) if input was (l, c)
        return out
    
class MLPDown(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer=4, latent_dim=4096, downsample=4):
        super().__init__()
        self.pooling = nn.Sequential(
            PoolingCNN(in_dim, downsample),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, latent_dim),
            nn.GELU(),
            *[nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU()) for _ in range(n_layer)],
            nn.Linear(latent_dim, out_dim)
        )
    
    def forward(self, x):
        x = self.pooling(x)
        return self.mlp(x)


class CompressionModel(pl.LightningModule):
    def __init__(self, cfg, gradio_progress=False, id_mapping=True):
        super().__init__()
        self.id_mapping = id_mapping

        self.downsample = 2

        self.compress = MLP(cfg.in_dim, cfg.mood_dim, cfg.n_layer, cfg.latent_dim)
        #self.uncompress = MLP(cfg.mood_dim, cfg.out_dim, cfg.n_layer, cfg.latent_dim)
        self.uncompress = MLPDown(cfg.mood_dim, cfg.out_dim, cfg.n_layer, cfg.latent_dim, downsample=self.downsample)
        if self.id_mapping:
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
        fg_masks = batch[2] # b, l
        feats_compressed = self.compress(feats) # b, l, c
        feats_uncompressed = self.uncompress(feats_compressed)

        # Downsample fg_masks to match the downsampled feature map size
        fg_masks_downsampled = rearrange(fg_masks[:, 1:], 'b (h w) -> b h w', h=int((fg_masks.shape[1]-1)**0.5), w=int((fg_masks.shape[1]-1)**0.5))
        fg_masks_downsampled = F.max_pool2d(fg_masks_downsampled.unsqueeze(1).float(), kernel_size=self.downsample, stride=self.downsample).squeeze(1).bool()
        fg_masks_downsampled = torch.cat([fg_masks[:, :1], fg_masks_downsampled.reshape(fg_masks.shape[0], -1)], dim=1)

        if self.id_mapping:
            feats_uncompressed_dummy = self.uncompress_dummy(feats_compressed)
        
        eigvec_gt, eigval_gt = ncut_wrapper(feats[fg_masks], self.cfg.n_eig)
        eigvec_hat, eigval_hat = ncut_wrapper(rearrange(feats_compressed, 'b l c -> (b l) c'), self.cfg.n_eig)
        eigvec_hat = eigvec_hat[fg_masks.flatten()]

        total_loss = 0
        if self.cfg.flag_loss > 0:
            compressed = feats_compressed[fg_masks]
            gt_sim = 0
            n_eig = 2
            n_sum = 0
            while n_eig <= self.cfg.n_eig:
                n_eig *= 2
                _eigvec = eigvec_gt[:, 1:n_eig]
                _eigvec = F.normalize(_eigvec, dim=-1)
                sim = _eigvec @ _eigvec.T
                gt_sim += sim
                n_sum += 1
            gt_sim /= n_sum
            hat_sim = compressed @ compressed.T
            # vmax, vmin, vmean, vstd = torch.max(gt_sim), torch.min(gt_sim), torch.mean(gt_sim), torch.std(gt_sim)
            # print(f"gt_sim: max={vmax:.4f}, min={vmin:.4f}, mean={vmean:.4f}, std={vstd:.4f}")
            # norms = torch.norm(compressed, dim=-1)
            # print(f"norms: max={norms.max():.4f}, min={norms.min():.4f}, mean={norms.mean():.4f}, std={norms.std():.4f}")
            flag_loss = F.smooth_l1_loss(gt_sim, hat_sim)
            self.log("loss/flag", flag_loss, prog_bar=True)
            total_loss += flag_loss * self.cfg.flag_loss
            self.loss_history['flag'].append(flag_loss.item())
        
        if self.cfg.eigvec_loss > 0:
            eigvec_loss = flag_space_loss(eigvec_gt, eigvec_hat, n_eig=self.cfg.n_eig)
            self.log("loss/eigvec", eigvec_loss, prog_bar=True)
            total_loss += eigvec_loss * self.cfg.eigvec_loss
            self.loss_history['eigvec'].append(eigvec_loss.item())

        if (self.cfg.recon_loss_fg > 0) and torch.any(fg_masks_downsampled):
            recon_loss_fg = F.smooth_l1_loss(target_feats[fg_masks_downsampled], feats_uncompressed[fg_masks_downsampled])
            self.log("loss/recon_fg", recon_loss_fg, prog_bar=True)
            total_loss += recon_loss_fg * self.cfg.recon_loss_fg
            self.loss_history['recon'].append(recon_loss_fg.item())

        if self.id_mapping and self.cfg.recon_loss_fg_dummy > 0 and torch.any(fg_masks):
            recon_loss_fg_dummy = F.smooth_l1_loss(feats[fg_masks], feats_uncompressed_dummy[fg_masks])
            self.log("loss/recon_fg_dummy", recon_loss_fg_dummy, prog_bar=True)
            total_loss += recon_loss_fg_dummy * self.cfg.recon_loss_fg_dummy

        if (self.cfg.recon_loss_bg > 0) and not torch.all(fg_masks_downsampled):
            recon_loss_bg = F.smooth_l1_loss(target_feats[~fg_masks_downsampled], feats_uncompressed[~fg_masks_downsampled])
            self.log("loss/recon_bg", recon_loss_bg, prog_bar=True)
            total_loss += recon_loss_bg * self.cfg.recon_loss_bg

        if self.id_mapping and self.cfg.recon_loss_bg_dummy > 0 and not torch.all(fg_masks):
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
            boundary_loss = compute_boundary_loss(rearrange(feats_compressed, 'b l c -> (b l) c'),)
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


# def train_compression_model(model, cfg: DictConfig, input_feats, target_feats, 
#                             plus_masks=None, devices=[0], compute_fg_mask=False):
#     free_memory()
#     b, l, c = input_feats.shape
#     if compute_fg_mask and plus_masks is None:
#         plus_masks = get_fg_mask(input_feats)
#     if plus_masks is None:
#         plus_masks = torch.ones((b*l)).bool()
#     plus_masks = plus_masks.flatten()
#     input_feats = input_feats.flatten(end_dim=-2)
#     target_feats = target_feats.flatten(end_dim=-2)

#     # logger = pl.loggers.TensorBoardLogger(cfg.log_dir, name=cfg.name)
#     trainer = pl.Trainer(max_steps=cfg.steps,
#                          gradient_clip_val=cfg.grad_clip_val,
#                          accelerator="gpu", 
#                          devices=devices,
#                          enable_checkpointing=False,
#                         #  logger=logger,
#     )
#     dataset = DatasetWithSimplices(input_feats, target_feats, plus_masks)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
#     trainer.fit(model, dataloader)

#     return trainer


def train_compression_model(model, cfg: DictConfig, input_feats, target_feats, 
                            plus_masks=None, devices=[0], compute_fg_mask=False):
    free_memory()
    
    b, l, c = input_feats.shape

    # Assuming no plus masks
    plus_masks = torch.ones((b, l)).bool()

    dataset = DatasetWithSimplices(input_feats, target_feats, plus_masks)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # logger = pl.loggers.TensorBoardLogger(cfg.log_dir, name=cfg.name)
    trainer = pl.Trainer(max_steps=cfg.steps,
                         gradient_clip_val=cfg.grad_clip_val,
                         accelerator="gpu", 
                         devices=devices,
                         enable_checkpointing=False,
                        # logger=logger,
    )
    trainer.fit(model, dataloader)

    return trainer
