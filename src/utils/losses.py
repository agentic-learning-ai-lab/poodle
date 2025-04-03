import logging
import omegaconf
import torch
from torch.nn import functional as F

from src.utils.flow import convert_to_coords, warp
from src.utils.ssim import weighted_ssim

ALPHA_2 = [0.5/(4**4), 0.5/(4**3), 0.5/(4**2), 0.5/4, 0.5]

log = logging.getLogger(__name__)

def weigh_losses(loss_dict, loss_weights):
    weighted_loss_dict = {}
    for key in loss_weights:
        if isinstance(loss_weights[key], omegaconf.listconfig.ListConfig):
            for i, val in enumerate(loss_weights[key]):
                key_i = f"{key}_{i}"
                if key_i in loss_dict:
                    weighted_loss_dict[key_i] = loss_dict[key_i] * val
        elif isinstance(loss_weights[key], float) or isinstance(loss_weights[key], int):
            if key in loss_dict:
                weighted_loss_dict[key] = loss_dict[key] * loss_weights[key]
        else:
            raise ValueError
    return weighted_loss_dict

def squared_mag(x):
    return (x ** 2).sum(dim=1, keepdim=True)

def compute_valid_mask(flow, is_flow=True):
    if is_flow:
        return (convert_to_coords(flow).abs() <= 1).all(dim=1, keepdim=True).float()
    else:
        return (flow.abs() <= 1).all(dim=1, keepdim=True).float()

def feature_regression_loss(source_feats, target_feats, flow, mask):
    warped_target_feats = warp(target_feats, flow)
    regression_loss = ((source_feats - warped_target_feats) ** 2 * mask)
    regression_loss = regression_loss.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)
    return regression_loss.mean()

# do not use occlusion masking, still use valid masking
# because only when consistency works, does the occlusion masking work.
# we can use the source image to cover for regions without a bijection, and thus won't spike the cycle loss.
def feature_cycle_consistency_loss(feature, flow_1, flow_2, mask_2):
    target_features_in_source = warp(feature, flow_1)
    target_features_back_in_target = warp(target_features_in_source, flow_2)
    cycle_loss = ((feature - target_features_back_in_target) ** 2 * mask_2).sum(dim=(1, 2, 3)) / (mask_2.sum(dim=(1, 2, 3)) + 1e-6)

    return cycle_loss.mean()

def smoothness_loss_v1(source, flow, smoothness_lambda=1, order=2):
    assert order >= 1, 'smoothness order must be at least 1'
    img_gx, img_gy = gradients(source, stride=order)
    weights_x = torch.exp(-torch.mean(torch.abs(smoothness_lambda * img_gx), axis=1, keepdims=True))
    weights_y = torch.exp(-torch.mean(torch.abs(smoothness_lambda * img_gy), axis=1, keepdims=True))

    flow_gx, flow_gy = gradients(flow)
    for _ in range(order - 1):
        flow_gx = gradients(flow_gx)[0]
        flow_gy = gradients(flow_gy)[1]
        
    smoothness_loss = (torch.mean(weights_x * torch.abs(flow_gx)) + torch.mean(weights_y * torch.abs(flow_gy)))
    return smoothness_loss

def smoothness_loss(source, flow, smoothness_lambda=1, order=2):
    img_gx, img_gy = image_grads(source, stride=order) # N, C, H, W
    weights_x = torch.exp(-torch.mean(torch.abs(smoothness_lambda * img_gx), axis=1, keepdims=True))
    weights_y = torch.exp(-torch.mean(torch.abs(smoothness_lambda * img_gy), axis=1, keepdims=True))
    
    flow_gx, flow_gy = image_grads(flow)
    for _ in range(order - 1):
        flow_gx = image_grads(flow_gx)[0]
        flow_gy = image_grads(flow_gy)[1]
    
    smoothness_loss = (torch.mean(weights_x * torch.abs(flow_gx)) + torch.mean(weights_y * torch.abs(flow_gy))) / 2
    return smoothness_loss

def image_grads(image_batch, stride=1):
  image_batch_gh = image_batch[:, :, stride:] - image_batch[:, :,  :-stride]
  image_batch_gw = image_batch[:, :, :, stride:] - image_batch[:, :, :, :-stride]
  return image_batch_gh, image_batch_gw

def gradients(tensor, stride=1):
    tensor_gy = tensor[:, :, stride:] - tensor[:, :, :-stride]
    tensor_gx = tensor[:, :, :, stride:] - tensor[:, :, :, :-stride]
    return tensor_gx, tensor_gy

def reconstruction_loss(source, target, flow, mask):
    warped_target = warp(target.detach(), flow)
    diff = (source - warped_target) * mask
    score, weights = weighted_ssim(source, warped_target, mask)
    l1_diff = diff.abs()
    l2_diff = diff ** 2
    ssim_ = (score * weights)
    l1_diff = l1_diff.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)
    l2_diff = l2_diff.sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)
    ssim_ = ssim_.sum(dim=(1, 2, 3)) / (weights.sum(dim=(1, 2, 3)) + 1e-6)
    
    return l1_diff.mean(), l2_diff.mean(), ssim_.mean()
    
def compute_metrics(x1, x2, forward_flows, backward_flows, gt_flow=None, gt_valid_flow_mask=None, gt_flow_reverse=None, alpha1=0.01, alpha2=0.5, scale_alpha2=False):
    metric_dict = {}
    for i in range(len(forward_flows) - 1):
        cur_alpha2 = alpha2 if not scale_alpha2 else ALPHA_2[i]
        metric_dict[f'mean_flow_{i}'] = torch.linalg.norm(forward_flows[i], ord=2, dim=1).mean()
    metric_dict[f'mean_flow_IMAGE'] = torch.linalg.norm(forward_flows[-1], ord=2, dim=1).mean()
    return metric_dict

def sup_loss(forward_flow, gt_flow, gt_flow_mask):
    loss_dict = {}
    loss_dict['supervised_multiscale_EPE'] = 0.
    loss_dict['supervised_EPE'] = 0.
    if gt_flow is not None:
        forward_flow_ = [x[:gt_flow.shape[0]] for x in forward_flow]
        sparse = (gt_flow_mask is not None)
        if len(forward_flow_) > 1:
            loss_dict['supervised_multiscale_EPE'] = supervised_multiscale_EPE(forward_flow_, gt_flow, mask=gt_flow_mask, mean=True) 
        loss_dict['supervised_EPE'] = EPE(forward_flow_[-1], gt_flow, mask=gt_flow_mask, sparse=sparse, mean=True) 
    return loss_dict

def supervised_loss(pred_flow, gt_flow=None, gt_valid_flow_mask=None):
    if gt_flow is None:
        return 0.
    if gt_valid_flow_mask is None:
        return (pred_flow - gt_flow).abs().mean()
    else:
        return ((pred_flow - gt_flow).abs() * gt_valid_flow_mask).mean()
    
def supervised_multiscale_EPE(network_output, target_flow, mask=None, weights=None, sparse=False, mean=False):
    def one_scale(output, target, sparse, flow_mask):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = sparse_max_pool(target, (h, w))
        else:
            target_scaled = F.interpolate(target, (h, w), mode='bilinear')
        if flow_mask is not None:
            flow_mask = F.interpolate(flow_mask.type(torch.float32), (h,w), mode='nearest').type(torch.bool)
        return EPE(output, target_scaled, mask=flow_mask, sparse=sparse, mean=mean)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None and mean is False:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
    elif weights is None and mean is True:
        weights = [1.] * len(network_output)
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse, flow_mask=mask)
    return loss

def EPE(input_flow, target_flow, mask=None, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        if mask is None:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = ~(target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        else:
            mask = mask.reshape(EPE_map.shape)

        EPE_map = EPE_map[mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size

def sparse_max_pool(input, size):
    '''Downsample the input by considering 0 values as invalid.

    Unfortunately, no generic interpolation mode can resize a sparse map correctly,
    the strategy here is to use max pooling for positive values and "min pooling"
    for negative values, the two results are then summed.
    This technique allows sparsity to be minized, contrary to nearest interpolation,
    which could potentially lose information for isolated data points.'''

    positive = (input > 0).float()
    negative = (input < 0).float()
    output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
    return output