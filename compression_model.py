"""
Neural Compression Model for Feature Space Learning

This module implements a compression model that learns to compress and decompress
image features while preserving their geometric and semantic properties using
normalized cuts (NCut) and various geometric losses.
"""

import gc
from collections import defaultdict
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from omegaconf import DictConfig
import gradio as gr

# NCut and geometric utilities
from ncut_pytorch.utils.gamma import find_gamma_by_degree_after_fps
from ncut_pytorch import ncut_fn, kway_ncut
from ncut_pytorch.ncuts.ncut_nystrom import _plain_ncut
from ncut_pytorch.utils.math import rbf_affinity

# Geometric loss functions
from riemann_curvature_loss import (
    compute_riemann_curvature_loss, 
    compute_boundary_loss, 
    compute_repulsion_loss,
    compute_axis_align_loss
)


# ===== Loss Functions =====

def compute_kway_ncut_loss(ground_truth_eigvec, predicted_eigvec, n_eig: int) -> torch.Tensor:
    """
    Compute k-way normalized cut loss between ground truth and predicted eigenvectors.
    
    Args:
        ground_truth_eigvec (torch.Tensor): Ground truth eigenvectors
        predicted_eigvec (torch.Tensor): Predicted eigenvectors
        n_eig (int): Number of eigenvectors to use
        
    Returns:
        torch.Tensor: Smooth L1 loss between Gram matrices
    """
    gt_subset = ground_truth_eigvec[:, :n_eig]
    pred_subset = predicted_eigvec[:, :n_eig]
    
    # Compute Gram matrices and compare them
    gt_gram = gt_subset @ gt_subset.T
    pred_gram = pred_subset @ pred_subset.T
    
    return F.smooth_l1_loss(gt_gram, pred_gram)


def compute_flag_space_loss(ground_truth_eigvec, predicted_eigvec, n_eig: int, 
                           start: int = 4, step_mult: int = 2) -> torch.Tensor:
    """
    Compute flag space loss by aggregating k-way NCut losses across multiple scales.
    
    Args:
        ground_truth_eigvec (torch.Tensor): Ground truth eigenvectors
        predicted_eigvec (torch.Tensor): Predicted eigenvectors
        n_eig (int): Maximum number of eigenvectors
        start (int): Starting number of eigenvectors
        step_mult (int): Multiplication factor for scaling
        
    Returns:
        torch.Tensor: Aggregated flag space loss
    """
    # Handle edge cases
    if torch.all(ground_truth_eigvec == 0) or torch.all(predicted_eigvec == 0):
        return torch.tensor(0.0, device=ground_truth_eigvec.device)
    
    total_loss = 0.0
    current_n_eig = start // step_mult
    
    while True:
        current_n_eig *= step_mult
        total_loss += compute_kway_ncut_loss(
            ground_truth_eigvec, predicted_eigvec, current_n_eig
        )
        
        # Stop if we exceed the available eigenvectors
        max_available = min(ground_truth_eigvec.shape[1], predicted_eigvec.shape[1])
        if current_n_eig > max_available:
            break
    
    return total_loss


def compute_ncut_eigenvectors(features: torch.Tensor, n_eig: int, gamma: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function to compute NCut eigenvectors using RBF affinity.
    
    Args:
        features (torch.Tensor): Input features
        n_eig (int): Number of eigenvectors to compute
        gamma (float): RBF kernel parameter
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Eigenvectors and eigenvalues
    """
    # eigvec, eigval = ncut_fn(features, n_eig=n_eig, gamma=gamma, track_grad=True, device=features.device)
    affinity = rbf_affinity(features, gamma=gamma)
    eigvec, eigval = _plain_ncut(affinity, n_eig=n_eig)
    return eigvec, eigval


# ===== Foreground Mask Generation =====

@torch.no_grad()
def generate_foreground_mask(image_embeds: torch.Tensor, num_clusters: int = 3) -> torch.Tensor:
    """
    Generate foreground mask using clustering on image embeddings.
    
    Assumes the center of the image contains foreground objects and corners contain background.
    
    Args:
        image_embeds (torch.Tensor): Image embeddings of shape (batch, length, channels)
        num_clusters (int): Number of clusters for segmentation
        
    Returns:
        torch.Tensor: Boolean foreground mask of shape (batch, length)
    """
    # Handle 2D input by adding batch dimension
    if image_embeds.dim() == 2:
        image_embeds = image_embeds.unsqueeze(0)
    
    batch_size, seq_len, channels = image_embeds.shape
    hw_size = int(np.sqrt(seq_len))
    
    # Remove CLS token and reshape for processing
    patch_embeds = image_embeds[:, 1:].reshape(batch_size * hw_size * hw_size, channels)
    
    # Compute NCut clustering
    gamma = find_gamma_by_degree_after_fps(patch_embeds, degree=0.1)
    eigenvectors, _ = ncut_fn(patch_embeds, n_eig=10, gamma=gamma, device='cuda')
    
    # Perform k-way clustering
    cluster_onehot = kway_ncut(eigenvectors[:, :num_clusters])
    cluster_indices = cluster_onehot.argmax(dim=-1)
    cluster_indices = cluster_indices.reshape(batch_size, hw_size, hw_size)
    
    # Determine foreground based on center vs corners
    center_clusters = cluster_indices[:, hw_size//2, hw_size//2]  # Center pixels
    corner_clusters = torch.cat([
        cluster_indices[:, 0, 0], cluster_indices[:, 0, -1],
        cluster_indices[:, -1, 0], cluster_indices[:, -1, -1]
    ])
    
    # Use mode to find dominant clusters
    center_mode = center_clusters.mode().values.item()
    
    # Create foreground mask
    fg_mask = cluster_indices == center_mode
    fg_mask = fg_mask.reshape(batch_size, hw_size * hw_size)
    
    # Add back CLS token (always considered foreground)
    cls_mask = torch.ones((batch_size, 1), device=fg_mask.device, dtype=torch.bool)
    fg_mask = torch.cat([cls_mask, fg_mask], dim=1)
    
    return fg_mask


# ===== Neural Network Components =====

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron with GELU activations.
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        num_layers (int): Number of hidden layers
        hidden_dim (int): Hidden layer dimension
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 4, hidden_dim: int = 4096):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        
        # Add hidden layers
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SpatialPoolingCNN(nn.Module):
    """
    CNN layer for spatial pooling of feature maps with support for sequence inputs.
    
    Handles inputs with CLS tokens and reshapes appropriately for 2D convolution.
    
    Args:
        num_channels (int): Number of input/output channels
        downsample_factor (int): Downsampling factor for pooling
    """
    
    def __init__(self, num_channels: int, downsample_factor: int = 4):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.conv = nn.Conv2d(
            num_channels, num_channels, 
            kernel_size=downsample_factor, 
            stride=downsample_factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass supporting both (batch, seq_len, channels) and (seq_len, channels) inputs.
        """
        # Handle input shape variations
        added_batch_dim = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            added_batch_dim = True
        elif x.dim() != 3:
            raise ValueError(f"Expected input shape (B, L, C) or (L, C), got {x.shape}")

        batch_size, seq_len, channels = x.shape
        
        if seq_len < 2:
            raise ValueError("Sequence length must be at least 2 (1 CLS token + 1 patch)")

        # Validate that seq_len-1 is a perfect square (for spatial arrangement)
        spatial_size = int(round((seq_len - 1) ** 0.5))
        if spatial_size * spatial_size != (seq_len - 1):
            raise ValueError(f"seq_len-1 must be perfect square. Got {seq_len-1}")

        # Separate CLS token and spatial features
        cls_tokens = x[:, :1, :]  # (B, 1, C)
        spatial_features = x[:, 1:, :]  # (B, H*W, C)
        
        # Reshape to 2D for convolution
        spatial_2d = rearrange(
            spatial_features, 'b (h w) c -> b c h w', 
            h=spatial_size, w=spatial_size
        )
        
        # Apply pooling
        pooled_features = self.conv(spatial_2d)
        
        # Reshape back to sequence format
        pooled_sequence = rearrange(pooled_features, 'b c h w -> b (h w) c')
        
        # Concatenate CLS token back
        output = torch.cat([cls_tokens, pooled_sequence], dim=1)

        # Remove batch dimension if it was added
        if added_batch_dim:
            output = output.squeeze(0)
        
        return output


class MLPWithSpatialPooling(nn.Module):
    """
    MLP with integrated spatial pooling for handling sequence data with spatial structure.
    
    Args:
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        num_layers (int): Number of hidden layers
        hidden_dim (int): Hidden layer dimension
        downsample_factor (int): Spatial downsampling factor
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 4, 
                 hidden_dim: int = 4096, downsample_factor: int = 4):
        super().__init__()
        
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            SpatialPoolingCNN(hidden_dim, downsample_factor),
            nn.GELU()
        ]
        
        # Add hidden layers
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ===== Main Compression Model =====

class CompressionModel(pl.LightningModule):
    """
    Neural compression model for learning compressed feature representations.
    
    This model compresses input features to a lower-dimensional "mood space" and
    then decompresses them back, while preserving geometric and semantic properties
    through various loss functions including NCut-based losses and geometric constraints.
    
    Args:
        config: Configuration object containing model hyperparameters
        enable_gradio_progress (bool): Whether to show progress in Gradio interface
        use_identity_mapping (bool): Whether to use identity mapping for reconstruction
    """
    
    def __init__(self, config: DictConfig, enable_gradio_progress: bool = False, 
                 use_identity_mapping: bool = True):
        super().__init__()
        
        # Store configuration
        self.config = config
        self.use_identity_mapping = use_identity_mapping
        self.downsample_factor = 1
        
        # Build encoder-decoder architecture
        self.encoder = MultiLayerPerceptron(
            config.in_dim, config.mood_dim, config.n_layer, config.latent_dim
        )
        
        self.decoder = MLPWithSpatialPooling(
            config.mood_dim, config.out_dim, config.n_layer, 
            config.latent_dim, self.downsample_factor
        )
        
        # Optional identity mapping decoder
        if self.use_identity_mapping:
            self.identity_decoder = MultiLayerPerceptron(
                config.mood_dim, config.in_dim, config.n_layer, config.latent_dim
            )
        
        # Training utilities
        self.loss_history = defaultdict(list)
        self.enable_gradio_progress = enable_gradio_progress
        if enable_gradio_progress:
            self.progress_tracker = gr.Progress()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder-decoder."""
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return reconstructed

    def training_step(self, batch, batch_idx):
        """Single training step with multiple loss components."""
        
        # Update progress if using Gradio
        if (self.enable_gradio_progress and 
            self.trainer.global_step % 10 == 0 and 
            self.trainer.global_step > 0 and
            self.loss_history['recon']):
            
            progress = self.trainer.global_step / self.config.steps
            recent_loss = self.loss_history['recon'][-1]
            self.progress_tracker(progress, desc=f"Training, loss = {recent_loss:.4f}")

        # Unpack batch
        input_features, target_features, fg_masks = batch
        
        # Forward pass
        compressed_features = self.encoder(input_features)
        reconstructed_features = self.decoder(compressed_features)
        
        # Prepare downsampled masks for spatial consistency
        downsampled_masks = self._downsample_masks(fg_masks)
        
        # Optional identity mapping
        if self.use_identity_mapping:
            identity_reconstructed = self.identity_decoder(compressed_features)
        
        # Compute NCut eigenvectors for geometric consistency
        # subsample fg_masks, reduce the number of nodes for speed up
        eigvec_masks = self._subsample_eigvec_masks(fg_masks)
        gt_eigenvectors = self._compute_ncut_eigenvectors(input_features, eigvec_masks)
        pred_eigenvectors = self._compute_ncut_eigenvectors(compressed_features, eigvec_masks)
        
        # Aggregate all loss components
        total_loss = self._compute_total_loss(
            input_features, target_features, compressed_features,
            reconstructed_features, identity_reconstructed if self.use_identity_mapping else None,
            fg_masks, downsampled_masks, gt_eigenvectors, pred_eigenvectors
        )
        
        self.log("loss/total", total_loss, prog_bar=True)
        return total_loss
    
    def _subsample_eigvec_masks(self, fg_masks: torch.Tensor, n_subsample: int = 500) -> torch.Tensor:
        with torch.no_grad():
            shape = fg_masks.shape
            fg_masks = fg_masks.flatten().cpu().numpy()
            subsampled_masks = np.zeros(fg_masks.shape, dtype=bool)
            non_zero_indices = np.where(fg_masks)[0]
            shuffled_indices = np.random.permutation(non_zero_indices.shape[0])
            non_zero_indices = non_zero_indices[shuffled_indices[:n_subsample]]
            subsampled_masks[non_zero_indices] = 1
            subsampled_masks = subsampled_masks.reshape(shape)
            return subsampled_masks
    
    def _downsample_masks(self, fg_masks: torch.Tensor) -> torch.Tensor:
        """Downsample foreground masks to match decoder output resolution."""
        batch_size = fg_masks.shape[0]
        spatial_size = int((fg_masks.shape[1] - 1) ** 0.5)
        
        # Reshape spatial part (excluding CLS token)
        spatial_masks = rearrange(
            fg_masks[:, 1:], 'b (h w) -> b h w', h=spatial_size, w=spatial_size
        )
        
        # Apply max pooling for downsampling
        downsampled_spatial = F.max_pool2d(
            spatial_masks.unsqueeze(1).float(), 
            kernel_size=self.downsample_factor, 
            stride=self.downsample_factor
        ).squeeze(1).bool()
        
        # Add back CLS token
        cls_tokens = fg_masks[:, :1]
        downsampled_masks = torch.cat([
            cls_tokens, 
            downsampled_spatial.reshape(batch_size, -1)
        ], dim=1)
        
        return downsampled_masks
    
    def _compute_ncut_eigenvectors(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute NCut eigenvectors for masked features."""
        masked_features = features[masks]
        if len(masked_features) > 0:
            gamma = masked_features.var(0).sum().item()
            # print(f"gamma: {gamma}")
            # gamma = 1000  
            eigenvectors, _ = compute_ncut_eigenvectors(
                masked_features, self.config.n_eig, gamma=gamma
            )
            return eigenvectors
        else:
            # Return zero tensor if no masked features
            return torch.zeros((1, self.config.n_eig), device=features.device)
    
    def _compute_total_loss(self, input_features, target_features, compressed_features,
                           reconstructed_features, identity_reconstructed, 
                           fg_masks, downsampled_masks, gt_eigenvectors, pred_eigenvectors):
        """Compute and aggregate all loss components."""
        total_loss = 0.0
        
        # Eigenvector preservation loss
        if self.config.eigvec_loss > 0:
            eigvec_loss = compute_flag_space_loss(
                gt_eigenvectors, pred_eigenvectors, n_eig=self.config.n_eig
            )
            self.log("loss/eigvec", eigvec_loss, prog_bar=True)
            total_loss += eigvec_loss * self.config.eigvec_loss
            self.loss_history['eigvec'].append(eigvec_loss.item())

        # Foreground reconstruction loss
        if self.config.recon_loss_fg > 0 and torch.any(downsampled_masks):
            fg_recon_loss = F.smooth_l1_loss(
                target_features[downsampled_masks], 
                reconstructed_features[downsampled_masks]
            )
            self.log("loss/recon_fg", fg_recon_loss, prog_bar=True)
            total_loss += fg_recon_loss * self.config.recon_loss_fg
            self.loss_history['recon'].append(fg_recon_loss.item())

        # Identity mapping foreground loss
        if (self.use_identity_mapping and 
            self.config.recon_loss_fg_dummy > 0 and 
            torch.any(fg_masks)):
            
            id_fg_loss = F.smooth_l1_loss(
                input_features[fg_masks], 
                identity_reconstructed[fg_masks]
            )
            self.log("loss/recon_fg_dummy", id_fg_loss, prog_bar=True)
            total_loss += id_fg_loss * self.config.recon_loss_fg_dummy

        # Background reconstruction loss
        if self.config.recon_loss_bg > 0 and not torch.all(downsampled_masks):
            bg_recon_loss = F.smooth_l1_loss(
                target_features[~downsampled_masks], 
                reconstructed_features[~downsampled_masks]
            )
            self.log("loss/recon_bg", bg_recon_loss, prog_bar=True)
            total_loss += bg_recon_loss * self.config.recon_loss_bg

        # Identity mapping background loss
        if (self.use_identity_mapping and 
            self.config.recon_loss_bg_dummy > 0 and 
            not torch.all(fg_masks)):
            
            id_bg_loss = F.smooth_l1_loss(
                input_features[~fg_masks], 
                identity_reconstructed[~fg_masks]
            )
            self.log("loss/recon_bg_dummy", id_bg_loss, prog_bar=True)
            total_loss += id_bg_loss * self.config.recon_loss_bg_dummy

        # Geometric losses on compressed features
        fg_compressed = compressed_features[fg_masks]
        
        if self.config.riemann_curvature_loss > 0:
            curvature_loss = compute_riemann_curvature_loss(fg_compressed)
            self.log("loss/riemann_curvature", curvature_loss, prog_bar=True)
            total_loss += curvature_loss * self.config.riemann_curvature_loss

        if self.config.axis_align_loss > 0:
            axis_loss = compute_axis_align_loss(fg_compressed)
            self.log("loss/axis_align", axis_loss, prog_bar=True)
            total_loss += axis_loss * self.config.axis_align_loss

        if self.config.repulsion_loss > 0:
            repulsion_loss = compute_repulsion_loss(fg_compressed)
            self.log("loss/repulsion", repulsion_loss, prog_bar=True)
            total_loss += repulsion_loss * self.config.repulsion_loss

        if self.config.boundary_loss > 0:
            flattened_compressed = rearrange(compressed_features, 'b l c -> (b l) c')
            boundary_loss = compute_boundary_loss(flattened_compressed)
            self.log("loss/boundary", boundary_loss, prog_bar=True)
            total_loss += boundary_loss * self.config.boundary_loss

        return total_loss
    
    def configure_optimizers(self):
        """Configure the optimizer."""
        return torch.optim.NAdam(self.parameters(), lr=self.config.lr)


# ===== Dataset and Training Utilities =====

class FeatureDataset(torch.utils.data.Dataset):
    """
    Dataset for feature compression training.
    
    Args:
        input_features (torch.Tensor): Input feature tensors
        target_features (torch.Tensor): Target feature tensors
        foreground_masks (torch.Tensor): Foreground mask tensors
    """
    
    def __init__(self, input_features: torch.Tensor, target_features: torch.Tensor, 
                 foreground_masks: torch.Tensor):
        self.input_features = input_features
        self.target_features = target_features
        self.foreground_masks = foreground_masks
    
    def __len__(self) -> int:
        return len(self.input_features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.input_features[idx], 
            self.target_features[idx], 
            self.foreground_masks[idx]
        )


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def train_compression_model(model: CompressionModel, 
                          config: DictConfig,
                          input_features: torch.Tensor,
                          target_features: torch.Tensor, 
                          foreground_masks: Optional[torch.Tensor] = None,
                          devices: List[int] = [0],
                          compute_fg_mask: bool = False) -> pl.Trainer:
    """
    Train the compression model with the given data.
    
    Args:
        model: The compression model to train
        config: Training configuration
        input_features: Input feature tensors
        target_features: Target feature tensors
        foreground_masks: Optional foreground masks
        devices: GPU devices to use
        compute_fg_mask: Whether to compute foreground masks automatically
        
    Returns:
        pl.Trainer: The trained PyTorch Lightning trainer
    """
    clear_gpu_memory()
    
    batch_size, seq_len, channels = input_features.shape

    # Use all-ones masks if none provided (treat everything as foreground)
    if foreground_masks is None:
        foreground_masks = torch.ones((batch_size, seq_len), dtype=torch.bool)

    # TODO: Compute foreground masks if required

    # Create dataset and dataloader
    dataset = FeatureDataset(input_features, target_features, foreground_masks)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_steps=config.steps,
        gradient_clip_val=config.grad_clip_val,
        accelerator="gpu", 
        devices=devices,
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=False  # Disable default logger
    )
    
    # Train the model
    trainer.fit(model, dataloader)
    
    return trainer
