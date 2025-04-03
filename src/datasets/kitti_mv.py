import glob
import os
import logging
import torch

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from src.datasets.bdd100k_flow import random_crop
from src.datasets.registry import register_dataset

log = logging.getLogger(__name__)

@register_dataset
def kitti_mv(data_dir, split='train', img_size=(370, 1224), photometric=True, delta_t=1):
    assert split in ['train'], "Only train split is supported for KITTI multiview extension"
    if split == 'train':
        def transforms_(img1, img2):
            img1, img2 = TF.to_tensor(img1), TF.to_tensor(img2)
            img1, img2 = TF.resize(img1, (384, 1280), antialias=None), TF.resize(img2, (384, 1280), antialias=None)
            img1, img2 = TF.resize(img1, img_size, antialias=None), TF.resize(img2, img_size, antialias=None)
            img1, img2 = random_crop(img1, img2, img_size[0], img_size[1], relative_offset=5)
            
            if photometric:
                # color augs
                images = torch.stack((img1, img2), dim=0)
                images = torch.roll(images, np.random.randint(0, 3), dims=1)
                if torch.rand(1) > 0.5:
                    images = images.flip(1)
                images = TF.adjust_hue(images, torch.rand(1)*0.5)
                img1_aug, img2_aug = images[0], images[1]
            else:
                img1_aug, img2_aug = img1, img2
            
            return {"source": img1_aug, "target": img2_aug, "source_unaug": img1, "target_unaug": img2}
        dataset = KittiMVDataset(data_dir, transforms=transforms_, delta_t=delta_t)
    return dataset
 
class KittiMVDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = "train", transforms=None, delta_t=1):
        assert mode in ["train", "test"]
        
        self.data_dir = data_dir
        self.mode = mode

        videos = list(set(['_'.join(x.split("_")[:-1]) for x in sorted(glob.glob(os.path.join(data_dir, f"{mode}ing", 'image_2','*')))]))
        self.videos = videos
        self.image_paths = []
        for video in videos:
            frames = sorted(glob.glob(video + "_*.png"))
            for frameA, frameB in zip(frames, frames[delta_t:]):
                self.image_paths.append((frameA, frameB))

        self.image_paths = np.array(self.image_paths).astype(np.string_)
        self._dataset_len = len(self.image_paths)
        self.transforms = transforms
       
    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, idx):
        img1 = Image.open(self.image_paths[idx][0].decode())
        img2 = Image.open(self.image_paths[idx][1].decode())

        return self.transforms(img1, img2)
