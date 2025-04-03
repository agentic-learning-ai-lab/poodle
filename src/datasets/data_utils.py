import torch

# from src.utils.flow import convert_to_coords
# from src.utils.losses import ALPHA_2, compute_occlusion_mask


def parse_batch(batch, dataset):
    raise NotImplementedError("This function is deprecated.")
    # if dataset in ["kitti_flow"]:
    #     source, target, gt_flow, gt_valid_flow_mask = batch
    #     gt_flow_reverse = None
    # elif dataset in ["kitti_raw", "flyingthings", "sintel_raw", "sintel"]:
    #     source, target = batch
    #     gt_flow, gt_valid_flow_mask, gt_flow_reverse = None, None, None
    # elif dataset in ["hd1k"]: 
    #     source, target, gt_flow = batch
    #     gt_flow_reverse, gt_valid_flow_mask = None, torch.ones_like(gt_flow[:, :1])
    # elif dataset in ["flyingchairs"]:
    #     source, target, gt_flow, gt_flow_reverse = batch
    #     gt_valid_flow_mask = compute_occlusion_mask(gt_flow, gt_flow_reverse, alpha2=ALPHA_2[-1]) * (convert_to_coords(gt_flow).abs() <= 1).all(dim=1, keepdim=True).float()
    # else:
    #     source, target = batch[:2]
    #     gt_flow, gt_valid_flow_mask, gt_flow_reverse = None, None, None
        
    # return source, target, gt_flow, gt_valid_flow_mask, gt_flow_reverse