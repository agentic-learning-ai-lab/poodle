from collections import defaultdict
import collections
import torch
from torch import nn
from torch.nn import functional as F
from src.utils.uflow_resampler import resample

XYS = {}

def upsample_flow(flow, scale=2.):
    return resize_flow(flow.type(torch.float32), flow.shape[-2]*scale, flow.shape[-1]*scale).type(flow.dtype)
    

def resize_flow(x, height, width):
    """resize flows and scale them based on the resize

    Args:
        x (torch.Tensor): flow field of (N, 2, H, W) or (2, H, W)
        height (int): height
        width (int): width

    Returns:
        _type_: _description_
    """
    assert x.shape[-3] == 2, f'Invalid shape: {x.shape}'
    x_ = F.interpolate(x.type(torch.float32), size=(int(height), int(width)), mode='bilinear', align_corners=False, antialias=False).type(x.dtype)
    scale = torch.tensor([width / x.shape[-1], height / x.shape[-2]], device=x.device)
    if len(x_.shape) == 4:
        scale = scale[None, :, None, None]
    else:
        scale = scale[:, None, None]
    return x_ * scale

def mode_resize_flow(x, height, width):
    assert x.shape[-3] == 2, f'Invalid shape: {x.shape}'
    x_scale, y_scale = width / x.shape[-1], height / x.shape[-2]
    p_h, p_w = int(1 / y_scale), int(1 / x_scale)
    scale = torch.tensor([x_scale, y_scale], device=x.device)
    if len(x.shape) == 4:
        scale = scale[None, :, None, None]
    else:
        scale = scale[:, None, None]
    x_ = x * scale # (N, 2, H, W)
    x_ = x_.reshape(-1, 2, height, p_h, width, p_w).permute(0, 1, 2, 4, 3, 5).flatten(4) # (N, 2, n_H, n_W, p_h * p_w)
    x_ = torch.mode(x_, dim=-1).values
    return x_ 

@torch.no_grad()
def mask_invalid(coords, for_grid_sample=False):
    """Mask coordinates outside of the image.

    Valid = 1, invalid = 0.

    Args:
        coords: a 4D float tensor of image coordinates.

    Returns:
        The mask showing which coordinates are valid.
    """
    H, W = coords.shape[-2:]
    coords_rank = len(coords.shape)
    if coords_rank != 4:
        raise NotImplementedError() 
    if for_grid_sample:
        assert coords.shape[1] == 2, f'Invalid shape: {coords.shape}'
        mask = torch.prod(((coords >= -1.) & (coords <= 1.)).type(coords.dtype), dim=1, keepdim=True)
    else:
        mask = torch.prod(((coords >= 0.) & (coords[:, :1] <= W-1) & (coords[:, 1:] <= H-1)).type(coords.dtype), dim=1, keepdim=True)
    return mask

def compute_occlusion_mask(forward_flow, backward_flow, return_diff=False, alpha1=0.01, alpha2=0.5):
    # NOTE: flow in pixels values, not [-1, 1] coords
    reversed_forward_flow = warp(backward_flow, forward_flow)
    forward_flow_diff = forward_flow + reversed_forward_flow

    fb_sq_diff = torch.sum((forward_flow + reversed_forward_flow)**2, dim=1, keepdim=True)
    fb_sum_sq = torch.sum((forward_flow**2 + reversed_forward_flow**2), dim=1, keepdim=True)

    # TODO Scale alpha2 based on resolution
    occlusion_mask_ = occlusion_mask = (fb_sq_diff > alpha1 * fb_sum_sq + alpha2).type(forward_flow.dtype)
    occlusion_mask = 1. - occlusion_mask_

    if return_diff:
        return occlusion_mask, forward_flow_diff
    return occlusion_mask

def compute_selfsup_mask(forward_flow, backward_flow, fb_sigma):
    h, w = forward_flow.shape[-2:]
    
    forward_warp = flow_to_warp(forward_flow)
    reversed_forward_flow = warp(backward_flow, forward_warp, convert_flow_to_coords=False)
    fb_sq_diff = torch.sum((forward_flow + reversed_forward_flow)**2, dim=1, keepdim=True)
    fb_consistency = torch.exp(-fb_sq_diff / (fb_sigma**2 * (h**2 + w**2)))
    
    valid_mask = mask_invalid(forward_warp)
    return fb_consistency * valid_mask

def flow_to_warp(flow, for_grid_sample=False):
    """Compute the warp from the flow field.

    Args:
        flow: tf.tensor representing optical flow.

    Returns:
        The warp, i.e. the endpoints of the estimated flow.
    """
    assert len(flow.shape) == 4
    # Construct a grid of the image coordinates.
    height, width = flow.shape[-2:]
    # i, j = torch.meshgrid(
    #     torch.linspace(0, height-1., height, device=flow.device),
    #     torch.linspace(0, width-1., width, device=flow.device),
    #     indexing='ij')
    
    x, y = torch.meshgrid(
        torch.linspace(0, width-1., width, device=flow.device, dtype=flow.dtype),
        torch.linspace(0, height-1., height, device=flow.device, dtype=flow.dtype),
        indexing='xy')
    grid = torch.stack((x, y), dim=0) # (2, H, W)

    # Potentially add batch dimension to match the shape of flow.
    if len(flow.shape) == 4:
        grid = grid[None].repeat(flow.shape[0], 1, 1, 1)

    # Add the flow field to the image grid.
    warp_ = grid + flow
    # warp_ = warp_.flip(1)
    if for_grid_sample:
        # torch only, convert grid back to [-1, 1]
        warp_ = torch.stack((
            (warp_[:, 0] / (width - 1.)) * 2. - 1.,
            (warp_[:, 1] / (height - 1.)) * 2. - 1.
        ), dim=1)
    
    return warp_

def convert_to_coords(flow, H=None, W=None):
    return flow_to_warp(flow, for_grid_sample=True)

def warp(x: torch.Tensor, flow: torch.Tensor, flow_mask: torch.Tensor=None,
         convert_flow_to_coords: torch.Tensor=True, mode: str='bilinear',
         grid_sample=False):
    if not grid_sample and (flow_mask is not None or mode != 'bilinear'): 
        raise NotImplementedError('flow_mask and mode != bilinear are not supported yet for resample. Use grid_sample instead.')
    if convert_flow_to_coords:
        warp_ = flow_to_warp(flow, for_grid_sample=grid_sample)
    else:
        warp_ = flow
    if grid_sample:
        (N, C, H, W) = x.shape
        # ones = torch.ones((N, 1, H, W), device=x.device)   
        # x = torch.cat((x, ones), dim=1) # N, C+1, H, W
        x = nn.functional.grid_sample(input=x, grid=warp_.permute(0, 2, 3, 1), mode=mode, padding_mode='zeros', align_corners=False)
        return x
        # mask = x[:, -1:, :, :]; mask[mask > 0.999] = 1.0; mask[mask < 1.0] = 0.0
        # if flow_mask is not None:
        #     if flow_mask.dtype is not torch.bool:
        #         flow_mask = flow_mask.type(torch.bool)
        #     mask[~flow_mask] = 0.0
        # return x[:, :-1, :, :] * mask
    else:
        return resample(x, warp_.permute(0, 2, 3, 1))

def normalize_features(feature_list, normalize, center, across_channels,
                       across_features):
    """Normalizes feature tensors (e.g., before computing the cost volume).

    Args:
        feature_list: list of tf.tensors, each with dimensions [b, h, w, c]
        normalize: bool flag, divide features by their standard deviation
        center: bool flag, subtract feature mean
        across_channels: bool flag, compute mean and std across channels
        across_features: bool flag, compute mean and std across images

    Returns:
        list, normalized feature_list
    """

    # Compute feature statistics.

    statistics = collections.defaultdict(list)
    axes = [-3, -2, -1] if across_channels else [-2, -1]
    for feature_image in feature_list:
        mean = torch.mean(feature_image, dim=axes, keepdim=True)
        variance = torch.var(feature_image, dim=axes, keepdim=True)
        statistics['mean'].append(mean)
        statistics['var'].append(variance)

    if across_features:
        statistics['mean'] = ([torch.mean(torch.cat(statistics['mean']))] *
                            len(feature_list))
        statistics['var'] = [torch.mean(torch.cat(statistics['var']))
                            ] * len(feature_list)

    statistics['std'] = [torch.sqrt(v + 1e-16) for v in statistics['var']]

    # Center and normalize features.

    if center:
        feature_list = [
            f - mean for f, mean in zip(feature_list, statistics['mean'])
        ]
    if normalize:
        feature_list = [f / std for f, std in zip(feature_list, statistics['std'])]

    return feature_list

def resize_flow(x, height, width):
    """resize flows and scale them based on the resize

    Args:
        x (torch.Tensor): flow field of (N, 2, H, W) or (2, H, W)
        height (int): height
        width (int): width

    Returns:
        _type_: _description_
    """
    assert x.shape[-3] == 2, f'Invalid shape: {x.shape}'
    x_ = F.interpolate(x, size=(int(height), int(width)), mode='bilinear', align_corners=True)
    scale = torch.tensor([width / x.shape[-1], height / x.shape[-2]], device=x.device)
    if len(x_.shape) == 4:
        scale = scale[None, :, None, None]
    else:
        scale = scale[:, None, None]
    return x_ * scale
