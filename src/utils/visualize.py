
import gc
import cv2
import numpy as np
import torch
from torchvision.io import write_png
from torchvision.utils import make_grid
from torchvision.transforms.functional import resize
from torch.utils import data as torchdata
import flow_vis

from src.utils.flow import flow_to_warp, mask_invalid, warp, compute_occlusion_mask

def visualize_predictions(fp, source, target, gt_flow, gt_flow_mask, pred_flow, pred_backward_flow, p1, p2, occlusion_mask, flow_diff, rescale=1., scale_to_gt=True):
    N, C, H, W = source.shape
    
    label_ls, image_ls = [], []
    # images, warps and abs_dirr
    t_to_s, s_to_t = warp(target, pred_flow[0]), warp(source, pred_backward_flow[0])
    image_ls.append(source), label_ls.append("source"); image_ls.append(t_to_s), label_ls.append("t->s")
    image_ls.append(torch.abs(t_to_s - source)), label_ls.append("absdiff\nt->s,s")
    image_ls.append(target), label_ls.append("target"); image_ls.append(s_to_t), label_ls.append("s->t")
    image_ls.append(torch.abs(s_to_t - target)), label_ls.append("absdiff\ns->t,t")
    image_ls.append(torch.abs(source - target)), label_ls.append("absdiff\ns,t")
    
    # flows
    gt_flow_img, gt_flow_scale = flow_to_img(gt_flow, return_scale=True)
    if not scale_to_gt: gt_flow_scale = None # disable if no longer scaling
    image_ls.append(gt_flow_img), label_ls.append("gtflow")
    image_ls.append(flow_to_img(pred_flow[0] * gt_flow_mask if gt_flow_mask is not None else 1, scale=gt_flow_scale)), label_ls.append("flow\nmasked")
    # gt flow diffs
    gt_flow_diff = (gt_flow - pred_flow[0]) * (gt_flow_mask if gt_flow_mask is not None else 1)
    image_ls.append(flow_to_img(gt_flow_diff, scale=gt_flow_scale)), label_ls.append("gtflow\n diff")   
    image_ls.append(flow_to_img(pred_flow[0], scale=gt_flow_scale)), label_ls.append(f"flow")
    image_ls.append(flow_to_img(pred_backward_flow[0], scale=gt_flow_scale)), label_ls.append(f"flow rev")
    
    #masks
    valid_mask = mask_invalid(flow_to_warp(pred_flow[0])); valid_occ_mask = (valid_mask * occlusion_mask)
    valid_mask, occlusion_mask, valid_occ_mask = (x.tile(1, 3, 1, 1) for x in [valid_mask, occlusion_mask, valid_occ_mask])
    image_ls.extend([flow_to_img(flow_diff, scale=gt_flow_scale), valid_mask, occlusion_mask, valid_occ_mask]); label_ls.extend(["flow diff", "valid msk", "occ msk", "valocc msk"]) 
        
    image_ls = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in image_ls]
    save_images(f"{fp}.png", *image_ls, row_prod=1, labels=label_ls, rescale=rescale)
    
    # image 2 for pyramid
    pyramid_ls, pyramid_label_ls = [], []
    pyramid_ls.extend([source,target]), pyramid_label_ls.extend(["source", "target"])
    for i in range(len(pred_flow)):
        f, b = pred_flow[i][:N], pred_backward_flow[i][:N]
        pyramid_ls.append(flow_to_img(f)), pyramid_label_ls.append(f"ff{i}")
        pyramid_ls.append(flow_to_img(b)), pyramid_label_ls.append(f"bf{i}")
    pyramid_ls = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in pyramid_ls]
    save_images(f"{fp}-pyramid-flow.png", *pyramid_ls, row_prod=1, labels=pyramid_label_ls, rescale=rescale)
    pyramid_ls, pyramid_label_ls = [], []
    pyramid_ls.extend([source,target]), pyramid_label_ls.extend(["source", "target"])
    for i in range(len(p1)):
        z1, z2 = p1[i][:N], p2[i][:N]
        pyramid_ls.append(z1.norm(dim=1, keepdim=True)), pyramid_label_ls.append(f"p1_{i}")
        pyramid_ls.append(z2.norm(dim=1, keepdim=True)), pyramid_label_ls.append(f"p2_{i}")
        pyramid_ls.append((z2-z1).norm(dim=1, keepdim=True)), pyramid_label_ls.append(f"p1_{i}-p2_{i}")
    pyramid_ls = [x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in pyramid_ls]
    save_images(f"{fp}-pyramid-feat.png", *pyramid_ls, row_prod=1, labels=pyramid_label_ls, rescale=rescale)

def flow_to_img(flow, scale=None, return_scale=False):
    is_torch = False
    if isinstance(flow, torch.Tensor):
        if flow.dtype is torch.bfloat16:
            flow = flow.type(torch.float32)
        is_torch = True
        flow = flow.detach().permute([0, 2, 3, 1] if flow.dim() == 4 else [1, 2, 0]).cpu().numpy()
    if scale is not None:
        scale = np.array(scale)
        assert return_scale is False, "Cannot return scale if scale is provided"
    if return_scale:
        flows, scales = [], []
        for x in flow:
            f, s = flow_to_color(x, convert_to_bgr=False, return_rad_max=True)
            flows.append(f); scales.append(s)
        flow = np.stack(flows, axis=0)
        scale = np.stack(scales, axis=0)
        if is_torch:
            return torch.from_numpy(flow).permute([0, 3, 1, 2] if len(flow.shape) == 4 else [2, 0, 1]).type(torch.float32)/255., torch.from_numpy(scale).type(torch.float32)
        return flow, scale
    else:
        if scale is not None:
            flow = np.stack([flow_to_color(x, scale=s, convert_to_bgr=False) for x,s in zip(flow, scale)], axis=0)
        else:
            flow = np.stack([flow_to_color(x, convert_to_bgr=False) for x in flow], axis=0)
        if is_torch:
            return torch.from_numpy(flow).permute([0, 3, 1, 2] if len(flow.shape) == 4 else [2, 0, 1]).type(torch.float32)/255.
        return  flow

def flow_to_color(flow_uv, scale=None, clip_flow=None, convert_to_bgr=False, return_rad_max=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad) if scale is None else scale
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    color_img = flow_vis.flow_uv_to_colors(u, v, convert_to_bgr)
    if return_rad_max:
        return color_img, rad_max
    else:
        return color_img

def get_mixed_batch(dataset, batch_size=12, seed=1):
    save_dataloader = torchdata.DataLoader(
        dataset=dataset,
        sampler=torchdata.RandomSampler(dataset, generator=torch.Generator().manual_seed(seed)),
        batch_size=batch_size, pin_memory=False,drop_last=True)
    iter = save_dataloader.__iter__()
    save_batch = iter.__next__()
    del iter
    del save_dataloader
    gc.collect()
    return save_batch

def make_image_grid(*args, row_prod=1, labels=[]):
    assert np.unique([x.shape[0] for x in args]).size == 1, f"Batch size must be the same for all inputs. {np.unique([x.shape[0] for x in args])}"
    N = args[0].shape[0]
    C, H, W = max([x.shape[-3] for x in args]), max([x.shape[-2] for x in args]), max([x.shape[-1] for x in args])
    combined = []
    for x in args:
        n, c, h, w = x.shape
        if c != C:
            if c == 1:
                x = x.tile(1, C, 1, 1)
            else: 
                raise ValueError(f"Channel dimension must be 1, got {c}.")
        if (h, w) != (H, W):
            x = resize(x, (H, W))
        combined.append(x.reshape(N, 1, C, H, W))
    combined = torch.cat(combined, dim=1).flatten(0, 1)
    if labels:
        label_imgs = []
        for label in labels:
            label_img = np.zeros((H, W, C), dtype=np.uint8)
            org = (int(0.10 * W), int(0.25 * H))
            fontscale = max(1, int(0.0125 * H))
            for i, line in enumerate(label.split('\n')):
                x, y = (org[0], org[1] + i * fontscale * 10)
                label_img = cv2.putText(img=label_img, text=line, org=(x, y), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=fontscale, color=(255, 255, 255), thickness=5)
            label_imgs.append(torch.from_numpy(label_img))
        label_imgs = torch.stack(label_imgs * row_prod, dim=0)
        label_imgs = label_imgs.permute(0, 3, 1, 2) / 255.
        combined = torch.cat([label_imgs, combined], dim=0)
    img_grid = make_grid((combined * 255.).type(torch.uint8), nrow=len(args) * row_prod)
    return img_grid
    
def save_images(fp, *args, row_prod=1, labels=[], rescale=1.):
    img_grid = make_image_grid(*args, row_prod=row_prod, labels=labels)
    img_grid = resize(img_grid, size=(int(img_grid.shape[-2] * rescale), int(img_grid.shape[-1] * rescale)), antialias=None)
    write_png(img_grid, fp)
