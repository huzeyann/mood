import torch
import torch.nn.functional as F
from ncut_pytorch.ncut_pytorch import find_gamma_by_degree_after_fps, nystrom_ncut
from ncut_pytorch.ncut_pytorch import affinity_from_features,ncut


import numpy as np
from einops import rearrange, repeat
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'monospace'


class EMA:
    def __init__(self, beta: float = 0.9):
        """
        Exponential Moving Average (EMA) implementation.
        
        Args:
            beta (float): Decay rate, should be in range (0,1).
        """
        self.beta = beta
        self.ema_value = None
    
    def update(self, value: torch.Tensor):
        """
        Updates the EMA with a new value.
        
        Args:
            value (torch.Tensor): The new value to incorporate.
        """
        if self.ema_value is None:
            self.ema_value = value.clone()
        else:
            self.ema_value.mul_(self.beta).add_(value * (1 - self.beta))
    
    def get_value(self):
        """
        Returns the current EMA value.
        
        Returns:
            torch.Tensor: The EMA value.
        """
        return self.ema_value
    
# def nystrom_ncut_fixed_sample_wrapper(features, n_eig, sample_idx, gamma=0.1, distance='rbf'):
#     eigvec, eigval = nystrom_ncut(features, n_eig, 
#                                 precomputed_sampled_indices=sample_idx,
#                                 distance=distance, affinity_focal_gamma=gamma,
#                                 indirect_connection=False, make_orthogonal=False)
#     return eigvec, eigval

def nystrom_ncut_fixed_sample_wrapper(features, n_eig, sample_idx, gamma=0.1, distance='rbf'):
    eigvec, eigval = nystrom_ncut(features, n_eig, 
                                precomputed_sampled_indices=np.arange(len(features)),
                                distance=distance, affinity_focal_gamma=gamma,
                                indirect_connection=False, make_orthogonal=False)
    # A = affinity_from_features(features, distance=distance, gamma=gamma)
    # eigvec, eigval = ncut(A, n_eig)
    return eigvec, eigval



def _eig_for_added_node(context, set_A, set_B, new_nodes, n_eig : int = 50, gamma=0.1, distance='rbf'):
    sample_idx = np.arange(len(context))
    n_eig = min(n_eig, int(context.shape[0] / 2 - 1))
    full_input = torch.cat([context, set_A, set_B, new_nodes], dim=0)
    eigvec, eigval = nystrom_ncut_fixed_sample_wrapper(full_input, n_eig, sample_idx, gamma=gamma, distance=distance)
    set_A_eigvec = eigvec[context.shape[0]:context.shape[0] + set_A.shape[0]]
    set_B_eigvec = eigvec[context.shape[0] + set_A.shape[0]:context.shape[0] + set_A.shape[0] + set_B.shape[0]]
    new_nodes_eigvec = eigvec[context.shape[0] + set_A.shape[0] + set_B.shape[0]:]
    return set_A_eigvec, set_B_eigvec, new_nodes_eigvec, eigval


def gradient_descent(context, set_A, set_B, new_nodes, n_eig,
                        weights_a, weights_b, n_interp,
                        max_iter, lr, distance, 
                        gamma, plot_loss=False, return_loss=False):

    ## Gradient descent to optimize the new nodes
    new_nodes = new_nodes.requires_grad_(True)
    optimizer = torch.optim.NAdam([new_nodes], lr=lr)
    losses = []
    ema_loss = EMA(beta=0.99)
    ema_values = []
    for i in range(max_iter):
        optimizer.zero_grad()
    
        set_A_eigvec, set_B_eigvec, new_nodes_eigvec, eigval = _eig_for_added_node(context, set_A, set_B, new_nodes, n_eig=n_eig, gamma=gamma, distance=distance)
        # compute ground truth eigvec
        set_A_eigvec = repeat(set_A_eigvec, "n_setA n_eig -> n_interp n_setA n_eig", n_interp=n_interp)
        set_B_eigvec = repeat(set_B_eigvec, "n_setB n_eig -> n_interp n_setB n_eig", n_interp=n_interp)
        target_eigvec = set_A_eigvec * weights_a[:, None, None] + set_B_eigvec * weights_b[:, None, None]
        target_eigvec = rearrange(target_eigvec, "n_interp n_setA n_eig -> (n_interp n_setA) n_eig")
        # compute loss
        loss = F.l1_loss(new_nodes_eigvec, target_eigvec)
        # flag loss
        # w = torch.arange(target_eigvec.shape[1], device=target_eigvec.device)
        # w = w.max() - w + 1
        # loss = F.l1_loss(new_nodes_eigvec * w, target_eigvec * w)
        loss.backward()
        optimizer.step()
        # for plotting
        losses.append(loss.item())
        ema_loss.update(loss)
        ema_values.append(ema_loss.get_value().item())
    new_nodes = rearrange(new_nodes, "(n_interp n_setA) D -> n_interp n_setA D", n_interp=n_interp)
    
    if plot_loss:
        plt.plot(losses, label="loss")
        plt.plot(ema_values, label="ema loss")
        plt.legend()
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("loss curve")
        plt.show()
    if return_loss:
        return new_nodes, losses
    return new_nodes


def interpolate_eigvec(context, set_A, set_B, n_eig : int = 50,
                        weights_a : torch.Tensor = None, weights_b : torch.Tensor = None,
                        max_iter=500, lr=0.001, distance='rbf', 
                        gamma_max_sample=1000, gamma=None, degree=0.1,
                        plot_loss=False, return_loss=False):

    """
    Interpolate the eigenvectors of the NCUT matrix between two sets of points.
    Args:
        context (torch.Tensor) shape [n_context, D]: Context points to compute the NCUT eigvectors.
        set_A (torch.Tensor) shape [n_setA, D]: First set of points. 
            Interpolation will be done between set_A and set_B, 
            e.g. 0.5 * set_A + 0.5 * set_B.
        set_B (torch.Tensor) shape [n_setB, D]: Second set of points. 
        n_eig (int): Number of eigenvectors to compute and interpolate.
        weights_a (torch.Tensor) shape [n_interp]: Weights for set_A.
            Interpolation is `batched`, multiple interpolation can be done at once.
            e.g. weight_a=[0.1, 0.5, 0.9], weight_b=[0.9, 0.5, 0.1]
            means return 3 interpolated points weighted by 0.1, 0.5, 0.9.
        weights_b (torch.Tensor) shape [n_interp]: Weights for set_B.
        max_iter (int): Maximum number of iterations for gradient descent, use plot_loss=True to plot the loss curve.
        lr (float): Learning rate for gradient descent.
        -----------
        n_eig: you need to change them to suit your data.
        lr, max_iter: check the loss curve and adjust them. use plot_loss=True to plot the loss curve.
        -----------
        -----------
        Other parameters: (you don't need to change them)
        -----------
        distance (str): Distance metric for NCUT, default is 'rbf'.
        gamma_max_sample (int): Number of samples to compute the gamma for NCUT.
        gamma (float): Gamma for NCUT, if None, will be computed from context.
        degree (float): Degree for NCUT, default is 0.1.
    Returns:
        new_nodes (torch.Tensor) shape [n_interp, n_setA, D]: Interpolated points.
            n_interp == len(weights_a) == len(weights_b) is the number of interpolations
            n_setA == len(set_A) == len(set_B) is the number of points in set_A and set_B.
    Examples:
        >>> context = torch.randn(1000, 2)
        >>> set_A = torch.randn(10, 2)
        >>> set_B = torch.randn(10, 2)
        >>> weight_a = torch.tensor([0.5])
        >>> weight_b = 1 - weight_a
        >>> new_nodes = interpolate_eigvec(context, set_A, set_B, n_eig=50,
        ...                                 weights_a=weight_a, weights_b=weight_b)
        >>> new_nodes.shape
        torch.Size([1, 10, 2])
        
        >>> weight_a = torch.tensor([0.1, 0.5, 0.9])
        >>> weight_b = 1 - weight_a
        >>> new_nodes = interpolate_eigvec(context, set_A, set_B, n_eig=50,
        ...                                 weights_a=weight_a, weights_b=weight_b)
        >>> new_nodes.shape
        torch.Size([3, 10, 2])
    """
    
    if gamma is None:
        with torch.no_grad():
            num_sample = min(gamma_max_sample, context.shape[0])
            gamma = find_gamma_by_degree_after_fps(context, degree=degree, num_sample=num_sample, distance=distance)
    
    
    # inilialize the new nodes
    n_interp = weights_a.shape[0]
    _setA = repeat(set_A, "n_setA D -> n_interp n_setA D", n_interp=n_interp)
    _setB = repeat(set_B, "n_setB D -> n_interp n_setB D", n_interp=n_interp)
    
    new_nodes = _setA * weights_a[:, None, None] + _setB * weights_b[:, None, None]
    new_nodes = rearrange(new_nodes, "n_interp n_setA D -> (n_interp n_setA) D")
    
    return gradient_descent(context, set_A, set_B, new_nodes, n_eig=n_eig,
                            weights_a=weights_a, weights_b=weights_b,
                            n_interp=n_interp, max_iter=max_iter, lr=lr,
                            distance=distance, gamma=gamma,
                            plot_loss=plot_loss, return_loss=return_loss)