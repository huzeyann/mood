# %%
import copy
from datetime import datetime
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from my_ipadapter_model import load_ipadapter, image_grid, generate
from my_intrinsic_dim import get_intrinsic_dim
from dino_clip_featextract import extract_dino_image_embeds, extract_dinov3_image_embeds, extract_clip_image_embeds, dino_img_transform, clip_img_transform, img_transform_inv
from gradio_utils import add_download_button
from my_dino_correspondence import get_correspondence_plot, ncut_tsne_multiple_images, kway_cluster_per_image, get_single_multi_discrete_rgbs, match_centers_three_images, match_centers_two_images, get_center_features
from compression_model_mkii import CompressionModel, train_compression_model, free_memory, get_fg_mask

import gradio as gr

USE_HUGGINGFACE_ZEROGPU = os.getenv("USE_HUGGINGFACE_ZEROGPU", "false")

if USE_HUGGINGFACE_ZEROGPU:  # huggingface ZeroGPU, dynamic GPU allocation 
    try:
        import spaces
    except:
        USE_HUGGINGFACE_ZEROGPU = False

import torch
from PIL import Image
import numpy as np
import skdim

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'monospace'

from omegaconf import OmegaConf


def train_mood_space(pil_images, lr=0.001, steps=5000, width=512, layers=4, dim=None, config_path="./config.yaml"): 
    images = load_gradio_images_helper(pil_images)
    dino_input_images = torch.stack([dino_img_transform(image) for image in images])
    clip_input_images = torch.stack([clip_img_transform(image) for image in images])
    dino_image_embeds = extract_dino_image_embeds(dino_input_images)
    clip_image_embeds = extract_clip_image_embeds(clip_input_images)

    if dim is None:
        dim = get_intrinsic_dim(dino_image_embeds.flatten(end_dim=-2))
        dim = int(dim)
        print(f"intrinsic dim is {dim}")
    else:
        print(f"using user-specified dim: {dim}")

    cfg = OmegaConf.load(config_path)
    cfg.mood_dim = dim
    cfg.lr = lr
    cfg.steps = steps
    cfg.latent_dim = width
    cfg.n_layer = layers

    model = CompressionModel(cfg, gradio_progress=True)  #TODO: check if gradio_progress works without gradio
    trainer = train_compression_model(model, cfg, dino_image_embeds, clip_image_embeds)
    return model, trainer

if USE_HUGGINGFACE_ZEROGPU:
    train_mood_space = spaces.GPU(duration=60)(train_mood_space)

def train_mood_space_visualize(image_embeds, dim=2, config_path="/workspace/n25c9900_2d.yaml"): 

    cfg = OmegaConf.load(config_path)
    cfg.mood_dim = dim
    cfg.in_dim = image_embeds.shape[-1]
    cfg.out_dim = image_embeds.shape[-1]

    model = CompressionModel(cfg, gradio_progress=True)  #TODO: check if gradio_progress works without gradio
    trainer = train_compression_model(model, cfg, image_embeds, image_embeds)
    return model, trainer



def load_gradio_images_helper(pil_images):
    if isinstance(pil_images[0], tuple):
        pil_images = [image[0] for image in pil_images]
    if isinstance(pil_images[0], str):
        pil_images = [Image.open(image) for image in pil_images]
    # convert to RGB
    pil_images = [image.convert("RGB") for image in pil_images]
    return pil_images


def find_direction_three_images(image_embeds, eigvecs, A2_to_A1, A1_to_B1):
    # image_embeds: b, l, c; b = 3, A2, A1, B1
    # eigvecs: b, l
    n_cluster = eigvecs[0].shape[-1]
    A1_center_features = get_center_features(image_embeds[1], eigvecs[1].argmax(-1).cpu(), n_cluster=n_cluster)
    B1_center_features = get_center_features(image_embeds[2], eigvecs[2].argmax(-1).cpu(), n_cluster=n_cluster)
    direction_A_to_B = []
    for i_A, i_B in enumerate(A1_to_B1):
        direction = B1_center_features[i_B] - A1_center_features[i_A]
        # direction = B1_center_features[i_B]
        # direction = direction / direction.norm(dim=-1, keepdim=True)
        direction_A_to_B.append(direction)
    direction_A_to_B = torch.stack(direction_A_to_B)

    cluster_labels = eigvecs[0].argmax(-1).cpu()
    n_cluster = eigvecs[0].shape[-1]
    direction_for_A2 = torch.zeros_like(image_embeds[0])
    for i_cluster in range(n_cluster):
        mask = cluster_labels == i_cluster
        if mask.sum() > 0:
            direction_for_A2[mask] = direction_A_to_B[A2_to_A1[i_cluster]]
    return direction_for_A2

def find_direction_two_images(image_embeds, eigvecs, A_to_B, unit_norm_direction=False):
    # image_embeds: A, B
    # eigvecs: A, B
    n_cluster = eigvecs[0].shape[-1]
    A_center_features = get_center_features(image_embeds[0], eigvecs[0].argmax(-1).cpu(), n_cluster=n_cluster)
    B_center_features = get_center_features(image_embeds[1], eigvecs[1].argmax(-1).cpu(), n_cluster=n_cluster)
    direction_A_to_B = []
    for i_A, i_B in enumerate(A_to_B):
        direction = B_center_features[i_B] - A_center_features[i_A]
        if unit_norm_direction:
            direction = direction / direction.norm(dim=-1, keepdim=True)
        direction_A_to_B.append(direction)
    direction_A_to_B = torch.stack(direction_A_to_B)

    cluster_labels = eigvecs[0].argmax(-1).cpu()
    n_cluster = eigvecs[0].shape[-1]
    direction_for_A = torch.zeros_like(image_embeds[0])
    for i_cluster in range(n_cluster):
        mask = cluster_labels == i_cluster
        if mask.sum() > 0:
            direction_for_A[mask] = direction_A_to_B[i_cluster]
    return direction_for_A

def analogy_three_images(image_list, model, ws, n_cluster=30, n_sample=1, match_method='hungarian'):
    # image_list: A2, A1, B1
    # ws: list of float
    # n_cluster: int
    # n_sample: int
    # match_method: str
    free_memory()
    images = torch.stack([dino_img_transform(image) for image in image_list])
    dino_image_embeds = extract_dino_image_embeds(images)
    compressed_image_embeds = model.compress(dino_image_embeds)
    input_embeds = dino_image_embeds
    _compressed_image_embeds = compressed_image_embeds
    original_images = images

    b, l, c = input_embeds.shape
    joint_eigvecs, joint_rgbs = ncut_tsne_multiple_images(input_embeds, n_eig=30, gamma=0.5)
    single_eigvecs = kway_cluster_per_image(input_embeds, n_cluster=n_cluster, gamma=0.5)
    # single_eigvecs = kway_cluster_multiple_images(input_embeds, n_cluster=n_cluster, gamma=0.5)
    discrete_rgbs = get_single_multi_discrete_rgbs(joint_rgbs, single_eigvecs)

    A2_to_A1, A1_to_B1 = match_centers_three_images(dino_image_embeds, single_eigvecs, match_method=match_method)

    direction = find_direction_three_images(_compressed_image_embeds, single_eigvecs, A2_to_A1, A1_to_B1)

    cluster_orders = [
        np.arange(n_cluster),
        A2_to_A1,
        A1_to_B1[A2_to_A1],
    ]
    correspondence_image = get_correspondence_plot(original_images, single_eigvecs, cluster_orders, discrete_rgbs, hw=16 * 2, n_cols=10)

    ip_model = load_ipadapter()
    
    n_steps = len(ws)
    interpolated_images = []
    fig, axs = plt.subplots(n_sample, n_steps, figsize=(n_steps * 2, n_sample * 3))
    axs = axs.flatten()
    progress = gr.Progress()
    for i_w, w in enumerate(ws):
        progress(i_w/n_steps, desc=f"Interpolating w={w:.2f}")
        A2_interpolated = _compressed_image_embeds[0] + direction * w
        A2_interpolated = model.uncompress(A2_interpolated)
        gen_images = generate(ip_model, A2_interpolated, num_samples=n_sample)
        interpolated_images.extend(gen_images)
        for i_img in range(n_sample):
            ax = axs[i_img * n_steps + i_w]
            ax.imshow(gen_images[i_img])
            ax.axis('off')
            if i_img == 0:
                ax.set_title(f"w={w:.2f}")
    fig.tight_layout()
    del ip_model
    free_memory()
    return correspondence_image, fig, interpolated_images

if USE_HUGGINGFACE_ZEROGPU:
    analogy_three_images = spaces.GPU(duration=60)(analogy_three_images)

def interpolate_two_images(image1, image2, model, ws, n_cluster=20, match_method='hungarian', unit_norm_direction=False, dino_matching=True, seed=None):
    free_memory()
    images = torch.stack([dino_img_transform(image) for image in [image1, image2]])
    dino_image_embeds = extract_dino_image_embeds(images)
    compressed_image_embeds = model.compress(dino_image_embeds)
    input_embeds = dino_image_embeds
    _compressed_image_embeds = compressed_image_embeds
    original_images = images

    b, l, c = input_embeds.shape
    joint_eigvecs, joint_rgbs = ncut_tsne_multiple_images(input_embeds, n_eig=30, gamma=0.5)
    single_eigvecs = kway_cluster_per_image(input_embeds, n_cluster=n_cluster, gamma=0.5)
    # single_eigvecs = kway_cluster_multiple_images(input_embeds, n_cluster=n_cluster, gamma=0.5)
    # discrete_rgbs = get_single_multi_discrete_rgbs(joint_rgbs, single_eigvecs)

    A_to_B = match_centers_two_images(dino_image_embeds[0], dino_image_embeds[1], single_eigvecs[0], single_eigvecs[1], match_method=match_method)

    if dino_matching:
        direction = find_direction_two_images(_compressed_image_embeds, single_eigvecs, A_to_B, unit_norm_direction=unit_norm_direction)
    else:
        direction = _compressed_image_embeds[1] - _compressed_image_embeds[0]

    ip_model = load_ipadapter()
    
    n_steps = len(ws)
    interpolated_images = []
    for i_w, w in enumerate(ws):
        A_interpolated = _compressed_image_embeds[0] + direction * w
        A_interpolated = model.uncompress(A_interpolated)

        # hw = int((A_interpolated.shape[0] - 1) ** 0.5)
        # A_interpolated_patch_tokens = rearrange(A_interpolated[1:, :], '(h w) c -> c h w', h=hw, w=hw)  # [C, 64, 64]
        # # Mean pool by factor 4 (64 -> 16)
        # A_interpolated_patch_tokens = F.avg_pool2d(A_interpolated_patch_tokens.unsqueeze(0), kernel_size=1, stride=1)
        # A_interpolated_patch_tokens = rearrange(A_interpolated_patch_tokens.squeeze(0), 'c h w -> (h w) c')
        # A_interpolated = torch.cat([A_interpolated[:1, :], A_interpolated_patch_tokens], dim=0)  # [257, 1280]

        gen_images = generate(ip_model, A_interpolated, num_samples=1, seed=seed)
        interpolated_images.extend(gen_images)
    
    del ip_model
    free_memory()
    return interpolated_images

if USE_HUGGINGFACE_ZEROGPU:
    interpolate_two_images = spaces.GPU(duration=60)(interpolate_two_images)

def plot_loss(model):
    # Plot loss curves from trainer
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    ax1.plot(model.loss_history['recon'])
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Reconstruction Loss')
    ax1.grid(True)

    eigvec_loss = - np.array(model.loss_history['eigvec'])
    ax2.plot(eigvec_loss)
    ax2.set_xlabel('Steps') 
    ax2.set_ylabel('Loss')
    ax2.set_title('Eigenvector Loss')
    ax2.grid(True)

    plt.tight_layout()
    
    return fig

DEFAULT_IMAGES_PATH = ["./images/black_bear1.jpg", "./images/black_bear2.jpg", "./images/pink_bear1.jpg"]
DEFAULT_IMAGES = [Image.open(image_path) for image_path in DEFAULT_IMAGES_PATH]
# DEFAULT_IMAGES = [image.resize((512, 512), resample=Image.Resampling.LANCZOS) for image in DEFAULT_IMAGES]
if USE_HUGGINGFACE_ZEROGPU:
    from download_models import download_ipadapter  
    download_ipadapter()
# %%
if __name__ == "__main__":
    import gradio as gr

    demo = gr.Blocks(
        theme=gr.themes.Base(spacing_size='md', text_size='lg', primary_hue='blue', neutral_hue='slate', secondary_hue='pink'),
    )
    with demo:
        model = gr.State([])

        with gr.Tab("1. Mood Space"):
            gr.Markdown("""
                        Instructions:
                        Please use the tabs to navigate through the app.
                        - Tab 1: Train a Mood Space compression model
                        - Tab 2: Interpolate between two images
                        - Tab 3: Path Lifting, given A1 -> B1, what's the A2 -> B2?
                        """)

            # gr.Markdown("Train a Mood Space compression model")

            with gr.Row():
                with gr.Column():
                    input_images = gr.Gallery(label="Mood Board Images", show_label=False)
                    upload_button = gr.UploadButton(elem_id="upload_button", label="Upload", variant='secondary', file_types=["image"], file_count="multiple")
                    
                    def convert_to_pil_and_append(images, new_images):
                        if images is None:
                            images = []
                        if new_images is None:
                            return images
                        if isinstance(new_images, Image.Image):
                            images.append(new_images)
                        if isinstance(new_images, list):
                            images += [Image.open(new_image) for new_image in new_images]
                        if isinstance(new_images, str):
                            images.append(Image.open(new_images))
                        gr.Info(f"Total images: {len(images)}")
                        return images
                    upload_button.upload(convert_to_pil_and_append, inputs=[input_images, upload_button], outputs=[input_images])
                    
                    # def load_example():
                    #     default_images = DEFAULT_IMAGES
                    #     return default_images
                    def load_images(images):
                        return images
                    # load_example_button = gr.Button("Load Example Images")
                    # load_example_button.click(load_example, inputs=[], outputs=input_images)
                    # add_download_button(input_images, filename_prefix="mood_board_images")

                with gr.Column():
                    with gr.Accordion("Training Parameters", open=False):
                        lr = gr.Slider(minimum=0.0001, maximum=0.01, step=0.0001, value=0.001, label="Learning Rate")
                        steps = gr.Slider(minimum=1000, maximum=100000, step=100, value=1500, label="Training Steps")
                        width = gr.Slider(minimum=16, maximum=4096, step=16, value=512, label="MLP Width")
                        layers = gr.Slider(minimum=1, maximum=8, step=1, value=4, label="MLP Layers")
                    train_button = gr.Button("Train", variant="primary")

                    def _train_wrapper(images, lr, steps, width, layers):
                        model, trainer = train_mood_space(images, lr, steps, width, layers)
                        loss_plot = plot_loss(model)
                        gr.Info(f"Training complete.")
                        return model, loss_plot

                    loss_plot = gr.Plot(label="Training Loss")
                    train_button.click(_train_wrapper, inputs=[input_images, lr, steps, width, layers], outputs=[model, loss_plot])

            example_groups = {
                "Dog -> Fish": ["./images/dog1.jpg", "./images/fish.jpg"],
                "Dog -> Paper": ["./images/dog1.jpg", "./images/paper2.jpg"],
                "Rotation": ["./images/black_bear1.jpg", "./images/black_bear2.jpg"],
                "Rotation (Analogy)": ["./images/black_bear1.jpg", "./images/black_bear2.jpg", "./images/pink_bear1.jpg"],
                "Duck -> Pixel": ["./images/duck1.jpg", "./images/duck_pixel.jpg"],
                "Duck -> Paper": ["./images/duck1.jpg", "./images/toilet_paper.jpg"],
                "Duck -> Paper (Analogy)": ["./images/duck1.jpg", "./images/toilet_paper.jpg", "./images/duck_pixel.jpg"],
            }

            def add_image_group_fn(group_gallery):
                images = [tup[0] for tup in group_gallery]
                # resize images to 512x512
                # images = [image.resize((512, 512), resample=Image.Resampling.LANCZOS) for image in images]
                return images
            
            gr.Markdown('## Examples')
            for group_name, group_images in example_groups.items():
                with gr.Row():
                    with gr.Column(scale=3):
                        add_button = gr.Button(value=f'add example [{group_name}]', elem_classes=['small-button'])
                    with gr.Column(scale=7):
                        group_gallery = gr.Gallery(
                            value=group_images,
                            columns=5,
                            rows=1,
                            height=200,
                            object_fit='scale-down',
                            label=group_name,
                            elem_classes=['large-gallery'],
                        )

                    add_button.click(
                        add_image_group_fn,
                        inputs=[group_gallery],
                        outputs=[input_images],
                    )

        with gr.Tab("2. Interpolate"):
            # gr.Markdown("Interpolate between two images")

            with gr.Row():
                input_A1 = gr.Image(label="A1", type="pil")
                input_B1 = gr.Image(label="B1", type="pil")
            
                with gr.Column():
                    # def _load_two_images():
                    #     default_images = DEFAULT_IMAGES[:2]
                    #     return default_images
                    # load_example_button3 = gr.Button("Load Example Images")
                    # load_example_button3.click(_load_two_images, inputs=[], outputs=[input_A1, input_B1])
                    fill_in_images_button = gr.Button("Reload Images")

                    with gr.Accordion("Interpolation Parameters", open=False):
                        w_left = gr.Slider(minimum=-10, maximum=10, step=0.01, value=0, label="Start w")
                        w_right = gr.Slider(minimum=-10, maximum=10, step=0.01, value=1, label="End w")
                        n_steps = gr.Slider(minimum=1, maximum=100, step=2, value=10, label="N interpolation")
                        n_sample = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="N samples per interpolation")
                        n_cluster = gr.Slider(minimum=1, maximum=100, step=1, value=10, label="N segments", info="for correspondence matching")
                        match_method = gr.Radio(choices=['hungarian', 'argmin'], value='hungarian', label="Matching Method")
                    interpolate_button = gr.Button("Run Interpolation", variant="primary")


            interpolated_images_plot = gr.Image(label="interpolated images")
            interpolated_images = gr.Gallery(label="Interpolated Images", show_label=False, visible=False)
            add_download_button(interpolated_images, filename_prefix="interpolated_images")

            def _infer_two_images(A1, B1, model, w_left, w_right, n_steps, n_cluster, n_sample, match_method):
                if model is None or model == []:
                    gr.Error("Please train a model first.")
                    return None, None, None
                pil_images = [A1, B1]
                images = load_gradio_images_helper(pil_images)
                ws = torch.linspace(w_left, w_right, n_steps)
                interpolated_images = interpolate_two_images(*images, model, ws, n_cluster, match_method)
                # resize interpolated_images to 512x512
                interpolated_images = [image.resize((512, 512), resample=Image.Resampling.LANCZOS) for image in interpolated_images]
                plot_images = [images[0].resize((512, 512), resample=Image.Resampling.LANCZOS)] + interpolated_images + [images[1].resize((512, 512), resample=Image.Resampling.LANCZOS)]
                plot_images = image_grid(plot_images, 2, len(plot_images)//2)
                return interpolated_images, plot_images
            interpolate_button.click(_infer_two_images, 
                                    inputs=[input_A1, input_B1, model, w_left, w_right, n_steps, n_cluster, n_sample, match_method], 
                                    outputs=[interpolated_images, interpolated_images_plot])

            ## fill in the images from input_images
            def fill_in_images(input_images):
                if input_images is None:
                    return None
                return input_images[0][0], input_images[1][0]
            fill_in_images_button.click(fill_in_images, inputs=[input_images], outputs=[input_A1, input_B1])
            input_images.change(fill_in_images, inputs=[input_images], outputs=[input_A1, input_B1])

        with gr.Tab("3. Path Lifting"):
            gr.Markdown("""                        
            given A1 -> B1, infer A2 -> B2
            """)

            with gr.Row():
                input_A1 = gr.Image(label="A1", type="pil")
                input_B1 = gr.Image(label="B1", type="pil")
                input_A2 = gr.Image(label="A2", type="pil")
                picked_B2 = gr.Image(label="B2", type="pil", interactive=False)
            
                with gr.Column():
                    # def _load_three_images():
                    #     default_images = DEFAULT_IMAGES
                    #     return default_images
                    # load_example_button2 = gr.Button("Load Example Images")
                    # load_example_button2.click(_load_three_images, inputs=[], outputs=[input_A2, input_A1, input_B1])
                    fill_in_images_button2 = gr.Button("Reload Images")
                    with gr.Accordion("Interpolation Parameters", open=False):
                        w_left = gr.Slider(minimum=-10, maximum=10, step=0.01, value=0, label="Start w")
                        w_right = gr.Slider(minimum=-10, maximum=10, step=0.01, value=1., label="End w")
                        n_steps = gr.Slider(minimum=1, maximum=100, step=2, value=12, label="N interpolation")
                        n_sample = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="N samples per interpolation")
                        n_cluster = gr.Slider(minimum=1, maximum=100, step=1, value=10, label="N segments", info="for correspondence matching")
                        match_method = gr.Radio(choices=['hungarian', 'argmin'], value='hungarian', label="Matching Method")
                    interpolate_button = gr.Button("Run Path Lifting", variant="primary")

                    def revert_images(A1, B1, A2, B2):
                        return B1, A1, B2, A2
                    revert_button = gr.Button("Revert Images", variant="secondary")
                    revert_button.click(revert_images, inputs=[input_A1, input_B1, input_A2, picked_B2], outputs=[input_A1, input_B1, input_A2, picked_B2])


            output_B2 = gr.Plot(label="B2 (interpolated)")
            interpolated_images = gr.Gallery(label="Interpolated Images", show_label=False, visible=True)
            correspondence_image = gr.Image(label="Correspondence Image", interactive=False)
            add_download_button(interpolated_images, filename_prefix="interpolated_images")

            def pick_best_image(interpolated_images, evt: gr.SelectData):
                best_image = interpolated_images[evt.index][0]
                logging_text = f"Selected Eigenvector at Index #{evt.index}"
                label = F'Eigenvector at Index #{evt.index}'
                return best_image
            interpolated_images.select(pick_best_image, interpolated_images, [picked_B2])

            def _infer_three_images(A2, A1, B1, model, w_left, w_right, n_steps, n_cluster, n_sample, match_method):
                if model is None or model == []:
                    gr.Error("Please train a model first.")
                    return None, None, None
                pil_images = [A2, A1, B1]
                images = load_gradio_images_helper(pil_images)
                ws = torch.linspace(w_left, w_right, n_steps)
                correspondence_image, fig, interpolated_images = analogy_three_images(images, model, ws, n_cluster, n_sample, match_method)
                # resize interpolated_images to 512x512
                interpolated_images = [image.resize((512, 512), resample=Image.Resampling.LANCZOS) for image in interpolated_images]
                return correspondence_image, fig, interpolated_images
            interpolate_button.click(_infer_three_images, 
                                    inputs=[input_A2, input_A1, input_B1, model, w_left, w_right, n_steps, n_cluster, n_sample, match_method], 
                                    outputs=[correspondence_image, output_B2, interpolated_images])
            
            ## fill in the images from input_images
            def fill_in_images(input_images):
                if input_images is None:
                    return None
                if len(input_images) == 2:
                    return input_images[0][0], input_images[1][0], input_images[0][0]
                elif len(input_images) == 3:
                    return input_images[0][0], input_images[1][0], input_images[2][0]
            fill_in_images_button2.click(fill_in_images, inputs=[input_images], outputs=[input_A1, input_B1, input_A2])
            input_images.change(fill_in_images, inputs=[input_images], outputs=[input_A1, input_B1, input_A2])

        with gr.Tab("3. Make Plot"):

            plot_button = gr.Button("Make Plot", variant="primary")

            gallery_fig = gr.Gallery(label="Gallery", show_label=False, type="filepath")
            add_download_button(gallery_fig, filename_prefix="output_images")

            def open_images(imgA1, imgB1, imgA2, imgB2):
                img_list = []
                for _img in [imgA1, imgB1, imgA2, imgB2]:
                    img = load_gradio_images_helper([_img])
                    img = img[0].resize((512, 512), resample=Image.Resampling.LANCZOS)
                    img_list.append(img)
                img_grid = image_grid(img_list[:4], 1, 4)
                img_list.append(img_grid)
                return img_list
            
            plot_button.click(open_images, inputs=[input_A1, input_B1, input_A2, picked_B2], outputs=[gallery_fig])


    demo.launch(share=True, server_port=6006)
