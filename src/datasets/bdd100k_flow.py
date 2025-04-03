from collections import defaultdict
import math
import os
from typing import Callable, List

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset
from torch.utils import data as torchdata
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from decord import VideoReader, cpu

from src.datasets.registry import register_dataset
from src.utils.affine import AffineWithGrid
from src.utils.misc import get_rank

def round_up_odd(x):
    return int(np.ceil(x / 2) * 2 + 1)

def random_crop(image1, image2, crop_height=None, crop_width=None, relative_offset=5):
    if crop_height == image1.shape[1] and crop_width == image1.shape[2]:
        return image1, image2
    
    orig_height = image1.shape[-2]
    orig_width = image2.shape[-1]
    
    scale = 1.
    scale = max(scale, crop_height / orig_height, crop_width / orig_width)
    new_height = math.ceil(orig_height * scale)
    new_width = math.ceil(orig_width * scale)
    
    image1 = tvF.resize(image1, (new_height, new_width), antialias=None)
    image2 = tvF.resize(image2, (new_height, new_width), antialias=None)
    max_h = new_height - crop_height + 1
    max_w = new_width - crop_width + 1
    h1, w1 = np.random.randint(0, max_h), np.random.randint(0, max_w)
    h2 = np.random.randint(max(h1 - relative_offset, 0), min(h1 + relative_offset + 1, max_h))
    w2 = np.random.randint(max(w1 - relative_offset, 0), min(w1 + relative_offset + 1, max_w))
    
    image1 = tvF.crop(image1, h1, w1, crop_height, crop_width)
    image2 = tvF.crop(image2, h2, w2, crop_height, crop_width)
    
    return image1, image2

@register_dataset
def bdd100k_flow(data_dir, meta_info_file, split, img_size=None,
                 timeofday_filter=None,
                 delta_t1=[6, 6],
                 delta_t2=[3, 15],
                 repeat_sample=None,
                 sample_idx=None,
                 initial_size=[1080, 1920],
                 initial_scale=[0.75, 1.25],
                 initial_crop_size=[512, 1024],
                 aff_angle=[-30., 30.], aff_trans=None, aff_scale=[0.5, 2.0], aff_shear=None,
                 bound_translate=True,
                 color_distortion_strength=1.0,
                 norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225],
                 subset=1):
    assert subset <= 1.0
    assert color_distortion_strength <= 1.0
    assert split == 'train', 'Only support train split for now'

    # transforms
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    if img_size is not None:
        resize = transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR, antialias=None)
    if initial_crop_size is not None:
        random_crop = transforms.RandomCrop(initial_crop_size)
    
    # affine transform
    affine_transform = AffineWithGrid(degrees=aff_angle, translate=aff_trans, scale=aff_scale, shear=aff_shear, interpolation=InterpolationMode.BILINEAR, bound_translate=bound_translate)

    # data augmentations
    s = color_distortion_strength
    h, w = img_size if img_size is not None else initial_crop_size
    gaussian_kernel = (round_up_odd(0.1*h), round_up_odd(0.1 * w))
    data_aug = transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(gaussian_kernel, sigma=(0.1, 2.0))], p=0.5),
        normalize
    ])
    
    def aug_fn(img1: torch.Tensor, img2: torch.Tensor, img3: torch.Tensor):
        img1 = tvF.to_tensor(img1)
        img2 = tvF.to_tensor(img2)
        img3 = tvF.to_tensor(img3)

        if initial_scale is not None:
            assert initial_size is not None, 'initial_size must be provided if initial_scale is provided'
            scale = float(torch.empty(1).uniform_(initial_scale[0], initial_scale[1]).item())
            img1, img2, img3 = torch.unbind(
                tvF.resize(torch.stack([img1, img2, img3], dim=0), (int(initial_size[0]*scale), int(initial_size[1]*scale)), interpolation=InterpolationMode.BILINEAR),
                dim=0
            )

        if initial_crop_size is not None:
            img1, img2, img3 = torch.unbind(
                random_crop(torch.stack([img1, img2, img3], dim=0)),
                dim=0
            )
        
        # flow learning augs
        flow_images = torch.stack((img1, img2), dim=0)
        flow_images = torch.roll(flow_images, np.random.randint(0, 3), dims=1)
        if torch.rand(1) > 0.5:
            flow_images = flow_images.flip(1)
        flow_images = tvF.adjust_hue(flow_images, torch.rand(1)*0.5)
        flow_img1_aug, flow_img2_aug = flow_images[0], flow_images[1]

        # flow-e augs
        ssl_img1_aug, f_aff1, b_aff1 = affine_transform(img1)
        ssl_img3_aug, f_aff3, b_aff3 = affine_transform(img3)

        f_aff1, b_aff1, f_aff3, b_aff3 = [x.permute(2, 0, 1) for x in [f_aff1, b_aff1, f_aff3, b_aff3]]

        if img_size is not None:
            flow_img1_aug, flow_img2_aug, ssl_img1_aug, ssl_img3_aug, b_aff1, b_aff3, img1, img2, img3 = \
                [resize(x) for x in [flow_img1_aug, flow_img2_aug, ssl_img1_aug, ssl_img3_aug, b_aff1, b_aff3, img1, img2, img3]]
        
        ssl_img1_aug = data_aug(ssl_img1_aug)
        ssl_img3_aug = data_aug(ssl_img3_aug)

        return {'source': flow_img1_aug,
                'target': flow_img2_aug,
                'ssl_source': ssl_img1_aug,
                'ssl_target': ssl_img3_aug,
                'ssl_source_baff': b_aff1,
                'ssl_target_baff': b_aff3,
                'source_unaug': img1,
                'target_unaug': img2,
                'ssl_target_unaug': img3}

    dataset = BDD100KFlowDataset(
        data_dir=data_dir,
        aug_fn=aug_fn,
        mode=split,
        meta_info_file=meta_info_file,
        timeofday_filter=timeofday_filter,
        delta_t1=delta_t1,
        delta_t2=delta_t2,
        norm_mean=norm_mean,
        norm_std=norm_std,
        repeat_sample=repeat_sample,
        sample_idx=sample_idx,
    )

    if subset != 1.0:
        n = int(len(dataset) * subset)
        data = torchdata.random_split(data, (n, len(data)-n), generator=torch.Generator().manual_seed(1))[0]

    return dataset

class BDD100KFlowDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 aug_fn: Callable,
                 mode: str = "train",
                 meta_info_file: str = None,
                 timeofday_filter: List[str] = None,
                 delta_t1: List[int] = [6, 6],
                 delta_t2: List[int] = [3, 15],
                 norm_mean: List[float] = [0.485, 0.456, 0.406],
                 norm_std: List[float] = [0.229, 0.224, 0.225],
                 repeat_sample=None,
                 sample_idx: List[int] = None,
                 ):
        
        assert mode in ["train", "val", "test"]
        
        self.data_dir = data_dir
        self.aug_fn = aug_fn
        self.mode = mode
        self.delta_t1 = delta_t1
        self.delta_t2 = delta_t2
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.repeat_sample = repeat_sample
        self.sample_idx = sample_idx

        self.max_delta_t = max([*delta_t1, *delta_t2])

        video_dir = os.path.join(data_dir, mode)
        video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir)]
        if meta_info_file is not None:
            meta_info = np.load(meta_info_file, allow_pickle=True).item()
            self.video_paths = []
            self.flow_dirs = []
            for p in video_paths:
                info = meta_info.get(p)
                if info is None:
                    continue
                length = info.get('length', 0)
                if sample_idx:
                    min_start = max(sample_idx)
                else:
                    min_start = 0
                if length <= (min_start + self.max_delta_t + 1):
                    continue
                if timeofday_filter is not None and info['attributes'].get('timeofday') not in timeofday_filter:
                    continue
                self.video_paths.append(p)
                self.flow_dirs.append(info.get('gt_flow_dir'))
        else:
            self.video_paths = video_paths
            self.flow_dirs = [None] * len(video_paths)
        self._dataset_len = len(self.video_paths)
       
    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        cpuid = 0 if worker_info == None else int(get_rank() * worker_info.num_workers + (worker_info.id))
        vr = VideoReader(self.video_paths[idx], num_threads=0, ctx=cpu(cpuid))
        if self.repeat_sample is not None:
            if self.sample_idx is not None:
                if len(self.sample_idx) > self.repeat_sample:
                    i_s = np.random.choice(self.sample_idx, size=self.repeat_sample, replace=False)
                elif len(self.sample_idx) == self.repeat_sample:
                    i_s = np.array(self.sample_idx)
                else:
                    raise ValueError(f"len(sample_idx) must be >= repeat_sample")
            else:
                i_s = np.random.randint(0, len(vr)-self.max_delta_t, size=self.repeat_sample)
            delta_t1s = np.random.randint(self.delta_t1[0], self.delta_t1[1]+1, size=self.repeat_sample)
            delta_t2s = np.random.randint(self.delta_t2[0], self.delta_t2[1]+1, size=self.repeat_sample)
            i_s = np.array([index for i, delta_t1, delta_t2 in zip(i_s, delta_t1s, delta_t2s) for index in [i, i+delta_t1, i+delta_t2]])
            sort_indexes = np.argsort(i_s).astype(np.int32)
            unsort_indexes = np.argsort(sort_indexes).astype(np.int32)

            imgs = vr.get_batch(list(i_s[sort_indexes])).asnumpy()[unsort_indexes]

            ls = defaultdict(list)
            for i in range(0, 3*self.repeat_sample, 3):
                res = self.aug_fn(imgs[i], imgs[i+1], imgs[i+2])
                for k, v in res.items():
                    ls[k].append(v)
            return {k: torch.stack(v, dim=0) for k, v in ls.items()}
        else:
            if self.sample_idx is not None:
                i = np.random.choice(self.sample_idx)
            else:
                i = np.random.randint(0, len(vr)-self.max_delta_t)
            delta_t1 = np.random.randint(self.delta_t1[0], self.delta_t1[1]+1)
            delta_t2 = np.random.randint(self.delta_t2[0], self.delta_t2[1]+1)
            img1 = vr[i].asnumpy()
            img2 = vr[i+delta_t1].asnumpy()
            img3 = vr[i+delta_t2].asnumpy()
            return self.aug_fn(img1, img2, img3)
