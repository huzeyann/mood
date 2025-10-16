"""
Mood Space Interactive Demo Application

This Gradio application provides an interactive interface for training and using
Mood Space compression models. The app includes three main functionalities:

1. Train a Mood Space compression model from uploaded images
2. Interpolate between two images using the trained model
3. Perform path lifting (analogy) given A1->B1, infer A2->B2

The application uses DINO/DINOv3 features for geometric understanding and
CLIP features for semantic representation, combined with neural compression
to learn a meaningful "mood space" representation.
"""

import copy
import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple, Union

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

# Local imports
from compression_model import CompressionModel, train_compression_model, clear_gpu_memory
from dino_correspondence import (
    get_correspondence_plot, ncut_tsne_multiple_images, 
    kway_cluster_per_image, get_discrete_colors_from_clusters, 
    match_centers_three_images, match_centers_two_images, 
    get_cluster_center_features
)
from extract_features import (
    extract_dino_features, extract_dinov3_features, extract_clip_features,
    dino_image_transform, clip_image_transform, image_inverse_transform
)
from gradio_utils import add_download_button
from ipadapter_model import load_ip_adapter_model, create_image_grid, generate_images_from_clip_embeddings
from intrinsic_dim import estimate_intrinsic_dimension

# Configure matplotlib for consistent styling
plt.rcParams['font.family'] = 'monospace'

# Configuration
USE_HUGGINGFACE_ZEROGPU = os.getenv("USE_HUGGINGFACE_ZEROGPU", "false").lower() == "true"
DEFAULT_IMAGES_PATH = ["./images/black_bear1.jpg", "./images/black_bear2.jpg", "./images/pink_bear1.jpg"]
DEFAULT_CONFIG_PATH = "./config.yaml"

# Optional HuggingFace Spaces integration
if USE_HUGGINGFACE_ZEROGPU:
    try:
        import spaces
    except ImportError:
        USE_HUGGINGFACE_ZEROGPU = False
        logging.warning("HuggingFace Spaces not available, running without GPU acceleration")


# ===== Utility Functions =====

def load_default_images() -> List[Image.Image]:
    """Load default example images for the demo."""
    try:
        return [Image.open(image_path) for image_path in DEFAULT_IMAGES_PATH]
    except Exception as e:
        logging.warning(f"Could not load default images: {e}")
        return []


def load_gradio_images_helper(pil_images: Union[List, Image.Image, str]) -> List[Image.Image]:
    """
    Convert various image input formats to a list of PIL Images.
    
    Args:
        pil_images: Images in various formats (PIL, paths, tuples)
        
    Returns:
        List[Image.Image]: Processed PIL images in RGB format
    """
    if pil_images is None:
        return []
    
    # Handle single image
    if isinstance(pil_images, Image.Image):
        return [pil_images.convert("RGB")]
    
    if isinstance(pil_images, str):
        return [Image.open(pil_images).convert("RGB")]
    
    # Handle list of images
    processed_images = []
    for image in pil_images:
        if isinstance(image, tuple):  # Gradio gallery format
            image = image[0]
        
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, Image.Image):
            pass  # Already PIL Image
        else:
            continue
        
        processed_images.append(image.convert("RGB"))
    
    return processed_images


# ===== Core Training Functions =====

def train_mood_space(pil_images: List[Image.Image], 
                    learning_rate: float = 0.001,
                    training_steps: int = 5000, 
                    mlp_width: int = 512,
                    mlp_layers: int = 4, 
                    mood_dimension: Optional[int] = None,
                    config_path: str = DEFAULT_CONFIG_PATH) -> Tuple[CompressionModel, object]:
    """
    Train a Mood Space compression model from input images.
    
    This function extracts DINO and CLIP features from the input images,
    estimates the intrinsic dimensionality if not provided, and trains
    a neural compression model to learn a meaningful embedding space.
    
    Args:
        pil_images: List of PIL Images for training
        learning_rate: Learning rate for optimization
        training_steps: Number of training steps
        mlp_width: Width of MLP layers
        mlp_layers: Number of MLP layers
        mood_dimension: Target dimensionality (auto-estimated if None)
        config_path: Path to configuration file
        
    Returns:
        Tuple of (trained_model, trainer)
    """
    # Process input images
    images = load_gradio_images_helper(pil_images)
    if len(images) == 0:
        raise ValueError("No valid images provided for training")
    
    # Transform images for feature extraction
    dino_input_images = torch.stack([dino_image_transform(image) for image in images])
    clip_input_images = torch.stack([clip_image_transform(image) for image in images])
    
    # Extract features using pre-trained models
    logging.info("Extracting DINO features...")
    dino_image_embeds = extract_dino_features(dino_input_images)
    
    logging.info("Extracting CLIP features...")
    clip_image_embeds = extract_clip_features(clip_input_images)
    
    # Determine target dimensionality
    if mood_dimension is None:
        flattened_features = dino_image_embeds.flatten(end_dim=-2)
        estimated_dim = estimate_intrinsic_dimension(flattened_features)
        mood_dimension = int(estimated_dim)
        logging.info(f"Estimated intrinsic dimension: {mood_dimension}")
    else:
        logging.info(f"Using user-specified dimension: {mood_dimension}")

    # Load and configure training parameters
    config = OmegaConf.load(config_path)
    config.mood_dim = mood_dimension
    config.lr = learning_rate
    config.steps = training_steps
    config.latent_dim = mlp_width
    config.n_layer = mlp_layers
    
    # Create and train model
    model = CompressionModel(config, enable_gradio_progress=True)
    trainer = train_compression_model(
        model, config, dino_image_embeds, clip_image_embeds
    )
    
    return model, trainer


def train_mood_space_for_visualization(image_embeds: torch.Tensor, 
                                     target_dim: int = 2,
                                     config_path: str = "/workspace/n25c9900_2d.yaml") -> Tuple[CompressionModel, object]:
    """
    Train a mood space model specifically for visualization purposes.
    
    Args:
        image_embeds: Pre-extracted image embeddings
        target_dim: Target dimensionality (typically 2 for visualization)
        config_path: Path to visualization-specific config
        
    Returns:
        Tuple of (trained_model, trainer)
    """
    config = OmegaConf.load(config_path)
    config.mood_dim = target_dim
    config.in_dim = image_embeds.shape[-1]
    config.out_dim = image_embeds.shape[-1]
    
    model = CompressionModel(config, enable_gradio_progress=True)
    trainer = train_compression_model(model, config, image_embeds, image_embeds)
    
    return model, trainer


# Apply HuggingFace Spaces GPU decorator if available
if USE_HUGGINGFACE_ZEROGPU:
    train_mood_space = spaces.GPU(duration=60)(train_mood_space)


# ===== Direction Finding Functions =====

def compute_direction_from_three_images(image_embeds: torch.Tensor, 
                                       eigenvectors: torch.Tensor,
                                       a2_to_a1_mapping: np.ndarray, 
                                       a1_to_b1_mapping: np.ndarray) -> torch.Tensor:
    """
    Compute direction vectors for three-image analogy (A2, A1, B1).
    
    Given the correspondence mappings, this function computes the direction
    from A1 to B1 and applies it to A2 to predict B2.
    
    Args:
        image_embeds: Image embeddings [A2, A1, B1]
        eigenvectors: Cluster eigenvectors for each image
        a2_to_a1_mapping: Cluster mapping from A2 to A1
        a1_to_b1_mapping: Cluster mapping from A1 to B1
        
    Returns:
        torch.Tensor: Direction field for A2
    """
    n_clusters = eigenvectors[0].shape[-1]
    
    # Compute cluster centers for A1 and B1
    a1_center_features = get_cluster_center_features(
        image_embeds[1], eigenvectors[1].argmax(-1).cpu(), n_clusters
    )
    b1_center_features = get_cluster_center_features(
        image_embeds[2], eigenvectors[2].argmax(-1).cpu(), n_clusters
    )
    
    # Compute direction vectors from A1 to B1
    direction_vectors = []
    for i_a1, i_b1 in enumerate(a1_to_b1_mapping):
        direction = b1_center_features[i_b1] - a1_center_features[i_a1]
        direction_vectors.append(direction)
    direction_vectors = torch.stack(direction_vectors)
    
    # Apply direction to A2 based on cluster assignments
    cluster_labels = eigenvectors[0].argmax(-1).cpu()
    direction_field = torch.zeros_like(image_embeds[0])
    
    for i_cluster in range(n_clusters):
        cluster_mask = cluster_labels == i_cluster
        if cluster_mask.sum() > 0:
            mapped_direction = direction_vectors[a2_to_a1_mapping[i_cluster]]
            direction_field[cluster_mask] = mapped_direction
    
    return direction_field


def compute_direction_from_two_images(image_embeds: torch.Tensor, 
                                    eigenvectors: torch.Tensor,
                                    a_to_b_mapping: np.ndarray, 
                                    use_unit_norm: bool = False) -> torch.Tensor:
    """
    Compute direction vectors for two-image interpolation.
    
    Args:
        image_embeds: Image embeddings [A, B]
        eigenvectors: Cluster eigenvectors [A, B]
        a_to_b_mapping: Cluster mapping from A to B
        use_unit_norm: Whether to normalize direction vectors
        
    Returns:
        torch.Tensor: Direction field for image A
    """
    n_clusters = eigenvectors[0].shape[-1]
    
    # Compute cluster centers
    a_center_features = get_cluster_center_features(
        image_embeds[0], eigenvectors[0].argmax(-1).cpu(), n_clusters
    )
    b_center_features = get_cluster_center_features(
        image_embeds[1], eigenvectors[1].argmax(-1).cpu(), n_clusters
    )
    
    # Compute direction vectors
    direction_vectors = []
    for i_a, i_b in enumerate(a_to_b_mapping):
        direction = b_center_features[i_b] - a_center_features[i_a]
        if use_unit_norm:
            direction = F.normalize(direction, dim=-1)
        direction_vectors.append(direction)
    direction_vectors = torch.stack(direction_vectors)
    
    # Apply direction based on cluster assignments
    cluster_labels = eigenvectors[0].argmax(-1).cpu()
    direction_field = torch.zeros_like(image_embeds[0])
    
    for i_cluster in range(n_clusters):
        cluster_mask = cluster_labels == i_cluster
        if cluster_mask.sum() > 0:
            direction_field[cluster_mask] = direction_vectors[i_cluster]
    
    return direction_field


# ===== Main Application Functions =====

def perform_three_image_analogy(image_list: List[Image.Image], 
                               model: CompressionModel,
                               interpolation_weights: List[float], 
                               n_clusters: int = 30,
                               n_samples: int = 1, 
                               match_method: str = 'hungarian') -> Tuple[Image.Image, plt.Figure, List[Image.Image]]:
    """
    Perform three-image analogy: given A2, A1, B1, predict A2 -> B2.
    
    Args:
        image_list: List of three images [A2, A1, B1]
        model: Trained compression model
        interpolation_weights: Interpolation weights for generation
        n_clusters: Number of clusters for correspondence matching
        n_samples: Number of samples to generate per weight
        match_method: Method for cluster matching ('hungarian' or 'argmin')
        
    Returns:
        Tuple of (correspondence_plot, interpolation_plot, generated_images)
    """
    clear_gpu_memory()
    
    # Prepare images and extract features
    images = torch.stack([dino_image_transform(image) for image in image_list])
    dino_image_embeds = extract_dino_features(images)
    compressed_image_embeds = model.encoder(dino_image_embeds)
    
    # Compute correspondences and clustering
    joint_eigenvectors, joint_colors = ncut_tsne_multiple_images(dino_image_embeds, n_eig=30, gamma=0.5)
    cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=0.5)
    discrete_colors = get_discrete_colors_from_clusters(joint_colors, cluster_eigenvectors)
    
    # Find cluster correspondences
    a2_to_a1_mapping, a1_to_b1_mapping = match_centers_three_images(
        dino_image_embeds, cluster_eigenvectors, match_method=match_method
    )
    
    # Compute direction field
    direction_field = compute_direction_from_three_images(
        compressed_image_embeds, cluster_eigenvectors, a2_to_a1_mapping, a1_to_b1_mapping
    )
    
    # Create correspondence visualization
    cluster_orders = [
        np.arange(n_clusters),
        a2_to_a1_mapping,
        a1_to_b1_mapping[a2_to_a1_mapping],
    ]
    correspondence_plot = get_correspondence_plot(
        images, cluster_eigenvectors, cluster_orders, discrete_colors, hw=16 * 4, n_cols=10
    )
    
    # Generate interpolated images
    ip_model = load_ip_adapter_model()
    n_steps = len(interpolation_weights)
    generated_images = []
    
    fig, axes = plt.subplots(n_samples, n_steps, figsize=(n_steps * 2, n_samples * 3))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    progress = gr.Progress() if 'gr' in globals() else None
    
    for i_w, weight in enumerate(interpolation_weights):
        if progress:
            progress(i_w / n_steps, desc=f"Interpolating w={weight:.2f}")
        
        # Interpolate in compressed space
        interpolated_embedding = compressed_image_embeds[0] + direction_field * weight
        decompressed_embedding = model.decoder(interpolated_embedding)
        
        # Generate images from interpolated embeddings
        batch_images = generate_images_from_clip_embeddings(
            ip_model, decompressed_embedding, num_samples=n_samples
        )
        generated_images.extend(batch_images)
        
        # Add to plot
        for i_sample in range(n_samples):
            ax = axes[i_sample, i_w]
            ax.imshow(batch_images[i_sample])
            ax.axis('off')
            if i_sample == 0:
                ax.set_title(f"w={weight:.2f}")
    
    fig.tight_layout()
    
    # Clean up
    del ip_model
    clear_gpu_memory()
    
    return correspondence_plot, fig, generated_images


def perform_two_image_interpolation(image1: Image.Image, 
                                  image2: Image.Image,
                                  model: CompressionModel, 
                                  interpolation_weights: List[float],
                                  n_clusters: int = 20, 
                                  match_method: str = 'hungarian',
                                  use_unit_norm: bool = False, 
                                  use_dino_matching: bool = True,
                                  seed: Optional[int] = None) -> List[Image.Image]:
    """
    Interpolate between two images using the trained compression model.
    
    Args:
        image1, image2: Input PIL Images
        model: Trained compression model
        interpolation_weights: Weights for interpolation
        n_clusters: Number of clusters for correspondence matching
        match_method: Method for cluster matching
        use_unit_norm: Whether to normalize direction vectors
        use_dino_matching: Whether to use DINO-based matching or simple interpolation
        seed: Random seed for generation
        
    Returns:
        List[Image.Image]: Generated interpolated images
    """
    clear_gpu_memory()
    
    # Prepare images and extract features
    images = torch.stack([dino_image_transform(img) for img in [image1, image2]])
    dino_image_embeds = extract_dino_features(images)
    compressed_image_embeds = model.encoder(dino_image_embeds)
    
    if use_dino_matching:
        # Use correspondence-based direction
        joint_eigenvectors, joint_colors = ncut_tsne_multiple_images(dino_image_embeds, n_eig=30, gamma=0.5)
        cluster_eigenvectors = kway_cluster_per_image(dino_image_embeds, n_clusters=n_clusters, gamma=0.5)
        
        a_to_b_mapping = match_centers_two_images(
            dino_image_embeds[0], dino_image_embeds[1],
            cluster_eigenvectors[0], cluster_eigenvectors[1], 
            match_method=match_method
        )
        
        direction_field = compute_direction_from_two_images(
            compressed_image_embeds, cluster_eigenvectors, a_to_b_mapping, use_unit_norm
        )
    else:
        # Simple linear interpolation in compressed space
        direction_field = compressed_image_embeds[1] - compressed_image_embeds[0]
    
    # Generate interpolated images
    ip_model = load_ip_adapter_model()
    generated_images = []
    
    for weight in interpolation_weights:
        interpolated_embedding = compressed_image_embeds[0] + direction_field * weight
        decompressed_embedding = model.decoder(interpolated_embedding)
        
        batch_images = generate_images_from_clip_embeddings(
            ip_model, decompressed_embedding, num_samples=1, seed=seed
        )
        generated_images.extend(batch_images)
    
    # Clean up
    del ip_model
    clear_gpu_memory()
    
    return generated_images


# Apply HuggingFace Spaces GPU decorator if available
if USE_HUGGINGFACE_ZEROGPU:
    perform_three_image_analogy = spaces.GPU(duration=60)(perform_three_image_analogy)
    perform_two_image_interpolation = spaces.GPU(duration=60)(perform_two_image_interpolation)


# ===== Visualization Functions =====

def plot_training_loss(model: CompressionModel) -> plt.Figure:
    """
    Create a plot showing training loss curves.
    
    Args:
        model: Trained compression model with loss history
        
    Returns:
        matplotlib.pyplot.Figure: Loss plot figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Reconstruction loss
    if 'recon' in model.loss_history and model.loss_history['recon']:
        ax1.plot(model.loss_history['recon'], 'b-', linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Reconstruction Loss')
        ax1.set_title('Reconstruction Loss')
        ax1.grid(True, alpha=0.3)

    # Eigenvector loss (negated for better visualization)
    if 'eigvec' in model.loss_history and model.loss_history['eigvec']:
        eigvec_loss = -np.array(model.loss_history['eigvec'])
        ax2.plot(eigvec_loss, 'r-', linewidth=2)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Eigenvector Loss (negated)')
        ax2.set_title('Eigenvector Preservation Loss')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ===== Gradio Interface Setup =====

def create_gradio_interface():
    """Create and configure the main Gradio interface."""
    
    # Load default images
    DEFAULT_IMAGES = load_default_images()
    
    # Download models if using HuggingFace Spaces
    if USE_HUGGINGFACE_ZEROGPU:
        try:
            from download_models import download_ipadapter
            download_ipadapter()
        except ImportError:
            logging.warning("Could not import download_models")

    # Create Gradio interface
    theme = gr.themes.Base(
        spacing_size='md', 
        text_size='lg', 
        primary_hue='blue', 
        neutral_hue='slate', 
        secondary_hue='pink'
    )
    
    demo = gr.Blocks(theme=theme)
    
    with demo:
        # Shared state for trained model
        model_state = gr.State([])

        # ===== Tab 1: Model Training =====
        with gr.Tab("1. Mood Space Training"):
            gr.Markdown("""
            ## Mood Space Training
            
            Upload images to train a neural compression model that learns a meaningful 
            "mood space" representation. The model will learn to compress high-dimensional 
            visual features while preserving semantic and geometric relationships.
            
            **Instructions:**
            1. Upload 2-10 images that share some thematic relationship
            2. Adjust training parameters if needed
            3. Click "Train" to start the training process
            4. Use the trained model in the other tabs for interpolation and analogies
            """)

            with gr.Row():
                with gr.Column():
                    # Image upload interface
                    input_images = gr.Gallery(
                        label="Training Images", 
                        show_label=True,
                        columns=3,
                        rows=2,
                        height=400
                    )
                    
                    upload_button = gr.UploadButton(
                        label="Upload Images", 
                        variant='secondary', 
                        file_types=["image"], 
                        file_count="multiple"
                    )
                    
                    def add_uploaded_images(existing_images, new_images):
                        """Add newly uploaded images to the gallery."""
                        if existing_images is None:
                            existing_images = []
                        if new_images is None:
                            return existing_images
                        
                        # Process new images
                        if isinstance(new_images, list):
                            new_pil_images = [Image.open(img) for img in new_images]
                        else:
                            new_pil_images = [Image.open(new_images)]
                        
                        existing_images.extend(new_pil_images)
                        gr.Info(f"Added {len(new_pil_images)} images. Total: {len(existing_images)}")
                        return existing_images
                    
                    upload_button.upload(
                        add_uploaded_images, 
                        inputs=[input_images, upload_button], 
                        outputs=[input_images]
                    )

                with gr.Column():
                    # Training parameters
                    with gr.Accordion("Training Parameters", open=False):
                        lr_slider = gr.Slider(
                            minimum=0.0001, maximum=0.01, step=0.0001, value=0.001,
                            label="Learning Rate", info="Higher values train faster but may be unstable"
                        )
                        steps_slider = gr.Slider(
                            minimum=1000, maximum=100000, step=100, value=1000,
                            label="Training Steps", info="More steps = better quality but slower"
                        )
                        width_slider = gr.Slider(
                            minimum=16, maximum=4096, step=16, value=512,
                            label="MLP Width", info="Network capacity"
                        )
                        layers_slider = gr.Slider(
                            minimum=1, maximum=8, step=1, value=4,
                            label="MLP Layers", info="Network depth"
                        )
                    
                    train_button = gr.Button("Train Model", variant="primary", size="lg")
                    
                    # Training wrapper function
                    def training_wrapper(images, lr, steps, width, layers):
                        """Wrapper for training that handles UI feedback."""
                        if not images or len(images) < 2:
                            gr.Error("Please upload at least 2 images for training")
                            return None, None
                        
                        try:
                            model, trainer = train_mood_space(images, lr, steps, width, layers)
                            loss_plot = plot_training_loss(model)
                            gr.Info("Training completed successfully!")
                            return model, loss_plot
                        except Exception as e:
                            gr.Error(f"Training failed: {str(e)}")
                            return None, None

                    loss_plot = gr.Plot(label="Training Progress")
                    train_button.click(
                        training_wrapper,
                        inputs=[input_images, lr_slider, steps_slider, width_slider, layers_slider],
                        outputs=[model_state, loss_plot]
                    )

            # Example image sets
            gr.Markdown("## Example Image Sets")
            example_sets = {
                "Bears (Rotation)": ["./images/black_bear1.jpg", "./images/black_bear2.jpg"],
                "Bear Analogy": ["./images/black_bear1.jpg", "./images/black_bear2.jpg", "./images/pink_bear1.jpg"],
                "Dog → Fish": ["./images/dog1.jpg", "./images/fish.jpg"],
                "Duck → Paper": ["./images/duck1.jpg", "./images/toilet_paper.jpg"],
            }
            
            for set_name, image_paths in example_sets.items():
                with gr.Row():
                    with gr.Column(scale=2):
                        add_btn = gr.Button(f"Load {set_name}", size="sm")
                    with gr.Column(scale=8):
                        example_gallery = gr.Gallery(
                            value=image_paths,
                            columns=len(image_paths),
                            rows=1,
                            height=150,
                            show_label=False
                        )
                    
                    def load_example_set(gallery_images):
                        return [img[0] if isinstance(img, tuple) else img for img in gallery_images]
                    
                    add_btn.click(
                        load_example_set,
                        inputs=[example_gallery],
                        outputs=[input_images]
                    )

        # ===== Tab 2: Two-Image Interpolation =====
        with gr.Tab("2. Image Interpolation"):
            gr.Markdown("""
            ## Two-Image Interpolation
            
            Smoothly interpolate between two images using the trained Mood Space model.
            The model finds semantic correspondences between image regions and creates
            meaningful transitions.
            """)

            with gr.Row():
                image_a = gr.Image(label="Image A", type="pil")
                image_b = gr.Image(label="Image B", type="pil")
                
                with gr.Column():
                    reload_btn = gr.Button("Load from Training Set")
                    
                    with gr.Accordion("Interpolation Settings", open=False):
                        w_start = gr.Slider(minimum=-2, maximum=2, step=0.1, value=0, label="Start Weight")
                        w_end = gr.Slider(minimum=-2, maximum=2, step=0.1, value=1, label="End Weight")
                        n_steps = gr.Slider(minimum=3, maximum=20, step=1, value=10, label="Number of Steps")
                        n_clusters = gr.Slider(minimum=5, maximum=50, step=1, value=10, label="Clusters for Matching")
                        match_method = gr.Radio(
                            ["hungarian", "argmin"],
                            value="hungarian",
                            label="Matching Method"
                        )
                    
                    interpolate_btn = gr.Button("Interpolate", variant="primary")

            interpolation_result = gr.Gallery(
                label="Interpolation Results", 
                columns=5, 
                rows=2
            )
            
            # Interpolation function
            def run_interpolation(img_a, img_b, model, w_start, w_end, n_steps, n_clusters, match_method):
                if model is None or model == []:
                    gr.Error("Please train a model first")
                    return None
                
                if img_a is None or img_b is None:
                    gr.Error("Please provide both input images")
                    return None
                
                weights = torch.linspace(w_start, w_end, n_steps).tolist()
                result_images = perform_two_image_interpolation(
                    img_a, img_b, model, weights, n_clusters, match_method
                )
                
                # Resize for display
                display_images = [
                    img.resize((256, 256), Image.Resampling.LANCZOS) 
                    for img in result_images
                ]
                
                return display_images
            
            interpolate_btn.click(
                run_interpolation,
                inputs=[image_a, image_b, model_state, w_start, w_end, n_steps, n_clusters, match_method],
                outputs=[interpolation_result]
            )
            
            # Auto-load from training images
            def load_first_two_images(training_images):
                if training_images and len(training_images) >= 2:
                    processed_images = load_gradio_images_helper(training_images)
                    if len(processed_images) >= 2:
                        return processed_images[0], processed_images[1]
                return None, None
            
            reload_btn.click(
                load_first_two_images,
                inputs=[input_images],
                outputs=[image_a, image_b]
            )

        # ===== Tab 3: Three-Image Analogy =====
        with gr.Tab("3. Visual Analogy"):
            gr.Markdown("""
            ## Visual Analogy (Path Lifting)
            
            Perform visual analogies: Given A1 → B1, predict A2 → B2.
            This demonstrates the model's ability to understand and transfer 
            visual transformations between different objects.
            """)

            with gr.Row():
                img_a1 = gr.Image(label="A1 (Source)", type="pil")
                img_b1 = gr.Image(label="B1 (Target)", type="pil") 
                img_a2 = gr.Image(label="A2 (Query)", type="pil")
                predicted_b2 = gr.Image(label="B2 (Predicted)", type="pil", interactive=False)
                
                with gr.Column():
                    reload_btn_3 = gr.Button("Load from Training Set")
                    
                    with gr.Accordion("Analogy Settings", open=False):
                        analogy_w_start = gr.Slider(minimum=-2, maximum=2, step=0.1, value=0, label="Start Weight")
                        analogy_w_end = gr.Slider(minimum=-2, maximum=2, step=0.1, value=1, label="End Weight") 
                        analogy_n_steps = gr.Slider(minimum=3, maximum=20, step=1, value=10, label="Number of Steps")
                        analogy_n_clusters = gr.Slider(minimum=5, maximum=50, step=1, value=10, label="Clusters for Matching")
                        analogy_match_method = gr.Radio(
                            ["hungarian", "argmin"],
                            value="hungarian",
                            label="Matching Method"
                        )
                    
                    analogy_btn = gr.Button("Run Analogy", variant="primary")

            # Analogy Results
            analogy_plot = gr.Plot(label="Analogy Results")
            
            # Correspondence Visualization
            correspondence_plot = gr.Image(label="Correspondence Visualization")
            
            analogy_results = gr.Gallery(
                label="Generated Sequence",
                columns=6,
                rows=2,
                visible=False
            )
            
            # Analogy function  
            def run_analogy(a1, b1, a2, model, w_start, w_end, n_steps, n_clusters, match_method):
                if model is None or model == []:
                    gr.Error("Please train a model first")
                    return None, None, None
                
                if a1 is None or b1 is None or a2 is None:
                    gr.Error("Please provide all three input images")
                    return None, None, None
                
                weights = torch.linspace(w_start, w_end, n_steps).tolist()
                correspondence_img, result_plot, result_images = perform_three_image_analogy(
                    [a2, a1, b1], model, weights, n_clusters, 1, match_method
                )
                
                return result_plot, correspondence_img, result_images
            
            analogy_btn.click(
                run_analogy,
                inputs=[img_a1, img_b1, img_a2, model_state, analogy_w_start, analogy_w_end, 
                       analogy_n_steps, analogy_n_clusters, analogy_match_method],
                outputs=[analogy_plot, correspondence_plot, analogy_results]
            )
            
            # Auto-load from training images
            def load_three_images(training_images):
                if training_images and len(training_images) >= 3:
                    processed_images = load_gradio_images_helper(training_images)
                    if len(processed_images) >= 3:
                        return processed_images[0], processed_images[1], processed_images[2]
                    elif len(processed_images) >= 2:
                        return processed_images[0], processed_images[1], processed_images[0]
                elif training_images and len(training_images) >= 2:
                    processed_images = load_gradio_images_helper(training_images)
                    if len(processed_images) >= 2:
                        return processed_images[0], processed_images[1], processed_images[0]
                return None, None, None
            
            reload_btn_3.click(
                load_three_images,
                inputs=[input_images],
                outputs=[img_a1, img_b1, img_a2]
            )

    return demo


# ===== Main Application Entry Point =====

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0" if USE_HUGGINGFACE_ZEROGPU else None,
        show_error=True
    )
