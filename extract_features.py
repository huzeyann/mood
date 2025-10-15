"""
Feature Extraction Module

This module provides utilities for extracting features from images using various
pre-trained models including DINO, DINOv3, and CLIP. It handles model loading,
batch processing, and memory management for efficient feature extraction.
"""

import gc
from typing import Tuple, Optional

import torch
import torch.nn as nn
from einops import rearrange
from torchvision import transforms

from ipadapter_model import extract_clip_embedding_tensor, load_ipadapter


# ===== Model URLs and Constants =====

DINOV3_MODEL_URLS = {
    "dinov3_vits16": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vits16plus": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "dinov3_vitb16": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vith16plus": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "dinov3_vitl16_sat493m": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    "dinov3_vitl16_dinotxt": "https://huggingface.co/huzey/mydv3/resolve/master/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
}

# Default hyperparameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_CHANNELS_TO_REMOVE = 24


# ===== Image Transforms =====

# High-resolution transform for DINO models
dino_image_transform = transforms.Compose([
    # transforms.Resize((256 * 4, 256 * 4)),  # High resolution for detailed features
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Standard resolution transform for CLIP models  
clip_image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standard ImageNet resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Inverse transform to convert normalized tensors back to PIL images
image_inverse_transform = transforms.Compose([
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    transforms.ToPILImage(),
])


# ===== Memory Management =====

def clear_gpu_memory():
    """Clear GPU cache and run garbage collection to free memory."""
    torch.cuda.empty_cache()
    gc.collect()


# ===== Model Classes =====

class DINOv3Backbone(nn.Module):
    """
    DINOv3 backbone model with channel variance filtering.
    
    This wrapper around DINOv3 models provides:
    - Automatic channel filtering based on variance
    - Consistent output format with CLS token + patch embeddings
    - Memory-efficient processing
    
    Args:
        model_config (str): DINOv3 model configuration name
        num_channels_to_remove (int): Number of high-variance channels to remove
    """
    
    def __init__(self, model_config: str = "dinov3_vitl16", 
                 num_channels_to_remove: int = DEFAULT_CHANNELS_TO_REMOVE):
        super().__init__()
        
        # Handle special case for satellite model
        if model_config == "dinov3_vitl16_sat493m":
            model_config = "dinov3_vitl16"
        
        # Load pre-trained DINOv3 model
        self.model = torch.hub.load(
            "facebookresearch/dinov3", 
            model_config, 
            weights=DINOV3_MODEL_URLS[model_config]
        )
        
        # Channel filtering parameters
        self.num_channels_to_remove = num_channels_to_remove
        self.channel_indices_to_keep = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv3 backbone.
        
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Features of shape (B, L, D) where L = 1 + H*W (CLS + patches)
        """
        # Extract patch embeddings and CLS tokens
        patch_embeddings, cls_tokens = self.model.get_intermediate_layers(
            x, return_class_token=True, reshape=True
        )[-1]  # Shape: (B, C, H, W), (B, C)
        
        # Initialize channel filtering on first forward pass
        if self.channel_indices_to_keep is None:
            self.channel_indices_to_keep = self._compute_channel_indices_to_keep(patch_embeddings)
        
        # Apply channel filtering
        filtered_patch_embeddings = patch_embeddings[:, self.channel_indices_to_keep, :, :]
        filtered_cls_tokens = cls_tokens[:, self.channel_indices_to_keep]
        
        # Reshape patch embeddings to sequence format
        patch_sequence = rearrange(filtered_patch_embeddings, 'b c h w -> b (h w) c')
        
        # Concatenate CLS token with patch embeddings
        output = torch.cat([filtered_cls_tokens.unsqueeze(1), patch_sequence], dim=1)
        
        return output
    
    def _compute_channel_indices_to_keep(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute channel indices to keep based on variance filtering.
        
        Removes channels with highest variance to reduce noise and dimensionality.
        
        Args:
            embeddings (torch.Tensor): Patch embeddings of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Indices of channels to keep
        """
        # Reshape to (N, C) for variance computation
        reshaped_embeddings = embeddings.permute(0, 2, 3, 1)  # (B, H, W, C)
        flattened_embeddings = reshaped_embeddings.reshape(-1, reshaped_embeddings.shape[-1])  # (N, C)
        
        # Compute variance across all spatial locations and batches
        channel_variances = torch.var(flattened_embeddings, dim=0)  # (C,)
        
        # Sort channels by variance (descending) and keep low-variance channels
        variance_sorted_indices = torch.argsort(channel_variances, descending=True)
        indices_to_keep = variance_sorted_indices[self.num_channels_to_remove:]
        
        return indices_to_keep


# ===== Feature Extraction Functions =====

@torch.no_grad()
def extract_dino_features(images: torch.Tensor, batch_size: int = DEFAULT_BATCH_SIZE) -> torch.Tensor:
    """
    Extract features using DINO ViT-S/16 model.
    
    Args:
        images (torch.Tensor): Input images of shape (N, C, H, W)
        batch_size (int): Batch size for processing
        
    Returns:
        torch.Tensor: DINO features of shape (N, L, D)
    """
    # Load DINO model
    #dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    dino_model = dino_model.eval().cuda()

    # Process images in batches
    num_batches = (images.shape[0] + batch_size - 1) // batch_size
    feature_batches = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, images.shape[0])
        
        batch_images = images[start_idx:end_idx].cuda()
        batch_features = dino_model.get_intermediate_layers(batch_images)[-1]
        feature_batches.append(batch_features.cpu())
    
    # Concatenate all batches
    all_features = torch.cat(feature_batches, dim=0)
    
    # Clean up memory
    del dino_model
    clear_gpu_memory()

    return all_features


@torch.no_grad()
def extract_dinov3_features(images: torch.Tensor, 
                           model_config: str = "dinov3_vitl16",
                           batch_size: int = DEFAULT_BATCH_SIZE) -> torch.Tensor:
    """
    Extract features using DINOv3 model with variance-based channel filtering.
    
    Args:
        images (torch.Tensor): Input images of shape (N, C, H, W)
        model_config (str): DINOv3 model configuration
        batch_size (int): Batch size for processing
        
    Returns:
        torch.Tensor: DINOv3 features of shape (N, L, D) where L = 1 + H*W
    """
    # Load DINOv3 backbone
    dinov3_backbone = DINOv3Backbone(model_config=model_config).eval().cuda()

    # Process images in batches
    num_batches = (images.shape[0] + batch_size - 1) // batch_size
    feature_batches = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, images.shape[0])
        
        batch_images = images[start_idx:end_idx].cuda()
        batch_features = dinov3_backbone(batch_images)
        feature_batches.append(batch_features.cpu())
    
    # Concatenate all batches
    all_features = torch.cat(feature_batches, dim=0)
    
    # Clean up memory
    del dinov3_backbone
    clear_gpu_memory()

    return all_features


@torch.no_grad()
def extract_clip_features(images: torch.Tensor, batch_size: int = DEFAULT_BATCH_SIZE) -> torch.Tensor:
    """
    Extract features using CLIP vision encoder.
    
    Args:
        images (torch.Tensor): Input images of shape (N, C, H, W)
        batch_size (int): Batch size for processing
        
    Returns:
        torch.Tensor: CLIP features of shape (N, L, D)
    """
    # Load IP-Adapter model (contains CLIP encoder)
    ip_adapter_model = load_ipadapter()
    
    # Process images in batches
    num_batches = (images.shape[0] + batch_size - 1) // batch_size
    feature_batches = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, images.shape[0])
        
        batch_images = images[start_idx:end_idx].cuda()
        batch_features = extract_clip_embedding_tensor(
            batch_images, ip_adapter_model, resize=False
        )
        feature_batches.append(batch_features.cpu())
    
    # Concatenate all batches
    all_features = torch.cat(feature_batches, dim=0)
    
    # Clean up memory
    del ip_adapter_model
    clear_gpu_memory()

    return all_features


# ===== Legacy Function Aliases =====

# Maintain backward compatibility with existing code
extract_dino_image_embeds = extract_dino_features
extract_dinov3_image_embeds = extract_dinov3_features  
extract_clip_image_embeds = extract_clip_features

# Legacy transform aliases
dino_img_transform = dino_image_transform
clip_img_transform = clip_image_transform
img_transform_inv = image_inverse_transform

