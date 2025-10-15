import numpy as np
import torch
import skdim

from ncut_pytorch.ncut_pytorch import farthest_point_sampling

import logging

def get_intrinsic_dim(feats, max_sample=2000):
    
    if isinstance(feats, torch.Tensor):
        feats = feats.cpu().detach().numpy()
        
    feats = torch.tensor(feats)
    feats = feats.reshape(-1, feats.shape[-1])
    
    if feats.shape[0] > max_sample:
        sample_idx = farthest_point_sampling(feats, max_sample)
        feats = feats[sample_idx]
    data = feats.cpu().numpy()
    
    id_est = skdim.id.MLE().fit(data)
    
    dim = id_est.dimension_
    
    if dim == 0:
        dim = np.mean(id_est.dimension_pw_)
        logging.warning(f"failed to estimate global intrinsic dimension, using average of local intrinsic dimension {dim}")
    
    return dim