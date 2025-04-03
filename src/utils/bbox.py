import torch
from torchvision.transforms import functional as tvF

def bbox_to_mask(bbox, h, w):
    """in torch compute binary mask of size h, w from batch of bounding box values """
    mask = torch.zeros((bbox.shape[0], h, w), dtype=torch.float32, device=bbox.device)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = x1.int(), y1.int(), x2.int(), y2.int()
        mask[i, y1:y2, x1:x2] = 1.
    return mask[:, None]

def resize_crop_to_bounding_boxes(image, bboxes, crop_size, bbox_index=None, mode=tvF.InterpolationMode.BILINEAR):
    N, num_crops = bboxes.shape[:2]
    bbox_size= bboxes[:, :, 2:] - bboxes[:, :, :2]
    return torch.stack([
        torch.stack([
            tvF.resized_crop(image[i, bbox_index[i, j]] if bbox_index is not None else image[i], 
                             top=int(bboxes[i, j, 1]), left=int(bboxes[i, j, 0]), #x1
                             height=int(bbox_size[i, j, 1]), width=int(bbox_size[i, j, 0]), 
                             size=crop_size, interpolation=mode) for j in range(num_crops)], dim=0) 
        for i in range(N)], dim=0)
