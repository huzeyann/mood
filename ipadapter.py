import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
torch.backends.cuda.enable_cudnn_sdp(False)  # a fix for torch 2.5.0

from ip_adapter import IPAdapterPlus
from ip_adapter import IPAdapter
# %%
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


@torch.inference_mode()
def extract_clip_embedding_pil(pil_image, ip_model):
    clip_image = ip_model.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
    clip_image = clip_image.to(ip_model.device, dtype=torch.float16)
    clip_image_embeds = ip_model.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
    clip_image_embeds = clip_image_embeds.float()
    return clip_image_embeds

def extract_clip_embedding_pil_batch(pil_images, ip_model):
    feats = []
    for image in pil_images:
        feats.append(extract_clip_embedding_pil(image, ip_model))
    feats = torch.cat(feats, dim=0)
    return feats

@torch.inference_mode()
def extract_clip_embedding_tensor(tensor_image, ip_model):
    tensor_image = tensor_image.to(ip_model.device, dtype=torch.float16)
    tensor_image = torch.nn.functional.interpolate(tensor_image, size=(224, 224), mode="bilinear", align_corners=False)
    clip_image_embeds = ip_model.image_encoder(tensor_image, output_hidden_states=True).hidden_states[-2]
    clip_image_embeds = clip_image_embeds.float()
    return clip_image_embeds


@torch.inference_mode()
def _myheck_ipadapter_get_image_embeds(self, pil_image=None, clip_image_embeds=None):
    if pil_image is not None:
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
    image_prompt_embeds = self.image_proj_model(clip_image_embeds)
    uncond_clip_image_embeds = self.image_encoder(
        torch.zeros(1, 3, 224, 224).to(self.device, dtype=torch.float16),
        output_hidden_states=True
    ).hidden_states[-2]
    uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
    return image_prompt_embeds, uncond_image_prompt_embeds


@torch.inference_mode()
def load_sdxl():

    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    # load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )
    return pipe

@torch.inference_mode()
def load_ipadapter():

    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "/data/IP-Adapter/models/image_encoder"
    ip_ckpt = "/data/IP-Adapter/models/ip-adapter-plus_sd15.bin"
    device = "cuda"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    # load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    # load ip-adapter
    ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    setattr(ip_model.__class__, "get_image_embeds", _myheck_ipadapter_get_image_embeds)
    
    return ip_model

@torch.inference_mode()
def load_ipadapter_global_embedding():

    # base_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    base_model_path = "runwayml/stable-diffusion-v1-5"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path = "/data/IP-Adapter/models/image_encoder"
    ip_ckpt = "/data/IP-Adapter/models/ip-adapter_sd15.bin"
    device = "cuda"

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
    # load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )
    # load ip-adapter
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
    
    return ip_model


@torch.inference_mode()
def generate(ip_model, clip_embeds, num_samples=4, num_inference_steps=50, seed=42):
    if clip_embeds.ndim == 2:
        clip_embeds = clip_embeds.unsqueeze(0)
    assert clip_embeds.ndim == 3
    assert clip_embeds.shape[0] == 1
    clip_embeds = clip_embeds.half().to(ip_model.device)
    images = ip_model.generate(clip_image_embeds=clip_embeds, pil_image=None,
        num_samples=num_samples, num_inference_steps=num_inference_steps, seed=seed)
    
    return images