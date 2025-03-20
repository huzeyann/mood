import torch
from scipy.spatial import Delaunay

# def _compute_delaunay_gpu(points):
#     """Compute Delaunay triangulation of points using CuPy (GPU-accelerated)"""
#     import cupy as cp 
#     from cupyx.scipy.spatial import Delaunay as CupyDelaunay
#     # Convert torch tensor to CuPy array
#     points_cp = cp.asarray(points.detach().cpu().numpy())
    
#     # Compute Delaunay triangulation on GPU
#     tri = CupyDelaunay(points_cp)
    
#     # Convert simplices back to torch tensor on the original device
#     simplices = torch.tensor(cp.asnumpy(tri.simplices), device=points.device)
#     return simplices

# @torch.no_grad()
# def compute_delaunay(points):
#     """Compute Delaunay triangulation of points"""
#     try:
#         return _compute_delaunay_gpu(points)
#     except ImportError:
#         return _compute_delaunay_cpu(points)

# from torch_delaunay.functional import shull2d

def pca_reduce_to_2d(points):
    u, s, v = torch.svd(points)
    return points @ v[:, :2]

@torch.no_grad()
def compute_delaunay(points):
    """Compute Delaunay triangulation of points"""
    points_2d = pca_reduce_to_2d(points)
    return Delaunay(points_2d.cpu().numpy()).simplices
    # return Delaunay(points.cpu().numpy()).simplices
  
def compute_riemann_curvature_loss(points, simplices=None, domain_min=0, domain_max=1):
    """
    Calculate loss based on approximated Riemann curvature.
    
    The loss measures deviations from uniform metric tensors across simplices,
    which approximates variations in Riemann curvature.
    """
    if simplices is None:
        simplices = compute_delaunay(points)
        
    ideal_det = torch.tensor(1.0, device=points.device, dtype=torch.float64)
    
    # Process each simplex in parallel 
    simplices_tensor = torch.tensor(simplices, device=points.device)
    
    # Extract points that form each simplex
    simplex_points = points[simplices_tensor]
    
    # Calculate edge vectors from the first point of each simplex
    edges = simplex_points[:, 1:] - simplex_points[:, 0].unsqueeze(1)
    
    # Compute metric tensors (Gram matrices) for each simplex
    metric_tensors = torch.matmul(edges, edges.transpose(1, 2))
    
    # Calculate determinants (related to volume distortion)
    dets = torch.linalg.det(metric_tensors)
    
    # Penalize deviations from constant determinant
    valid_dets = dets[dets > 0]
    total_curvature = torch.mean((valid_dets - ideal_det)**2)
    return total_curvature
    # # Add boundary repulsion to keep points inside domain
    # boundary_penalty = torch.mean(torch.relu(domain_min - points)) + \
    #                     torch.mean(torch.relu(points - domain_max))
    
    # # Add point repulsion term for additional stability
    # dist_matrix = torch.cdist(points, points)
    # # Set diagonal to large value to avoid self-repulsion
    # mask = torch.eye(points.shape[0], device=points.device).bool()
    # dist_matrix = dist_matrix + mask * 1000
    # repulsion = torch.mean(1.0 / (dist_matrix + 1e-8))
    
    # return total_curvature + 10.0 * boundary_penalty + 0.01 * repulsion


def compute_axis_align_loss(data):
    """ Encourage axis alignment by minimizing off-diagonal elements in the covariance matrix """
    n, d = data.shape
    centered_data = data - data.mean(dim=0)  # Center the data
    cov_matrix = (centered_data.T @ centered_data) / n  # Compute covariance matrix
    
    eye = torch.eye(d, device=data.device)
    return torch.nn.functional.smooth_l1_loss(cov_matrix, eye)  # L1 loss between covariance matrix and identity matrix


def compute_repulsion_loss(points):

    # Add point repulsion term for additional stability
    dist_matrix = torch.cdist(points, points)
    # Set diagonal to large value to avoid self-repulsion
    mask = torch.eye(points.shape[0], device=points.device).bool()
    dist_matrix = dist_matrix + mask * 1000
    repulsion = 1.0 / (dist_matrix + 1e-8)
    non_diag = repulsion[~mask]
    return torch.mean(non_diag)
    
def compute_boundary_loss(points, domain_min=-1, domain_max=1):
    return torch.mean(torch.relu(domain_min - points)) + \
            torch.mean(torch.relu(points - domain_max))