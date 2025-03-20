import gc
import torch
from ipadapter import extract_clip_embedding_tensor, load_ipadapter

def free_memory():
    torch.cuda.empty_cache()
    gc.collect()

@torch.no_grad()
def extract_dino_image_embeds(images, batch_size=32):
    dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    dino = dino.eval().cuda()

    num_batches = (images.shape[0] + batch_size - 1) // batch_size
    dino_image_embeds = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, images.shape[0])
        batch = images[start_idx:end_idx].cuda()
        batch_embeds = dino.get_intermediate_layers(batch)[-1]
        dino_image_embeds.append(batch_embeds)
    
    dino_image_embeds = torch.cat(dino_image_embeds, dim=0).cpu()
    
    del dino
    free_memory()

    return dino_image_embeds

@torch.no_grad()
def extract_clip_image_embeds(images, batch_size=32):
    ipmodel = load_ipadapter()
    num_batches = (images.shape[0] + batch_size - 1) // batch_size
    clip_image_embeds = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, images.shape[0])
        batch = images[start_idx:end_idx].cuda()
        batch_embeds = extract_clip_embedding_tensor(batch, ipmodel)
        clip_image_embeds.append(batch_embeds)
    
    clip_image_embeds = torch.cat(clip_image_embeds, dim=0).cpu()
    
    del ipmodel
    free_memory()

    return clip_image_embeds

from torchvision import transforms

img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_transform_inv = transforms.Compose([
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    transforms.ToPILImage(),
])

