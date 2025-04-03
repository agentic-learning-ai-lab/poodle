import logging

from torchvision.datasets import KittiFlow
import torchvision.transforms.functional as TF

from src.datasets.bdd100k_flow import random_crop
from src.datasets.registry import register_dataset

log = logging.getLogger(__name__)

@register_dataset
def kitti(data_dir, split='train', img_size=(370, 1224)):
    if split == 'train': 
        def transforms_(img1, img2, flow, valid_flow_mask):
            img1, img2 = TF.to_tensor(img1), TF.to_tensor(img2)
            img1, img2 = random_crop(img1, img2, img_size[0], img_size[1], relative_offset=5)
            return img1, img2, None, None
        dataset = KittiFlow(data_dir, split=split, transforms=transforms_)
    if split in ['val', 'eval']: # eval is the train set with flow labels
        if img_size not in [(370, 1224), None]:
            log.warn(f"img_size is not (370, 1224) or None. KITTI Flow EVAL is only cropped and will be to {img_size}")
        def transforms_(img1, img2, flow, valid_flow_mask):
            img1, img2 = TF.to_tensor(img1), TF.to_tensor(img2)
            flow = TF.to_tensor(flow.transpose(1, 2, 0))
            valid_flow_mask = TF.to_tensor(valid_flow_mask)
            if img_size is not None:
                h, w = img1.shape[-2:]
                offset_height = int((h - 370) / 2)
                offset_width = int((w- 1224) / 2)
                
                img1 = TF.crop(img1, offset_height, offset_width, img_size[0], img_size[1])
                img2 = TF.crop(img2, offset_height, offset_width, img_size[0], img_size[1])
                flow = TF.crop(flow, offset_height, offset_width, img_size[0], img_size[1])
                valid_flow_mask = TF.crop(valid_flow_mask, offset_height, offset_width, img_size[0], img_size[1])
            return img1, img2, flow, valid_flow_mask
    elif split in ['test']:
        raise NotImplementedError("KITTI Flow test/eval/val not implemented")

    if split in ['val', 'eval']:
        split = 'train'
    dataset = KITTIFlowDict(data_dir, split=split, transforms=transforms_)
    return dataset

class KITTIFlowDict(KittiFlow):
    def __getitem__(self, idx):
        img1, img2, flow, valid_flow_mask = super().__getitem__(idx)
        if flow == None and valid_flow_mask == None:
            return {"source": img1, "target": img2}
        else:
            return {"source": img1, "target": img2, "gt_flow": flow, "gt_flow_mask": valid_flow_mask}
 