import gc
import os
from typing import List

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as transformsv2
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import VideoReader as TVVideoReader
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.datasets.registry import register_dataset
from src.utils.affine import AffineWithGrid
from src.utils.ffprobe import ffprobe
from src.utils.misc import get_rank

import logging 
log = logging.getLogger(__name__)

@register_dataset
def bdd100k_flowe(data_dir,
                  meta_info_file,
                  split,
                  initial_size=(810, 1350),
                  crop_size=(512, 1024),
                  delta_t=[15, 30],
                  repeat_sample=1,
                  backend='decord'):
    assert split == 'train', 'Only support train split for now'
    assert backend in ['decord', 'torchvision', 'torchvision-videoreader']

    dataset = BDD100KFloweDataset(
        data_dir=data_dir,
        mode=split,
        meta_info_file=meta_info_file,
        delta_t=delta_t,
        initial_size=initial_size,
        crop_size=crop_size,
        repeat_sample=repeat_sample,
        backend=backend,
    )
    return dataset

class BDD100KFloweDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 mode: str = "train",
                 meta_info_file: str = None,
                 delta_t: List[int] = [15, 30],
                 initial_size: List[int] = [810, 1350],
                 crop_size: List[int] = [512, 1024],
                 repeat_sample=1,
                 backend='decord',
                 ):
        
        assert mode in ["train", "val", "test"]
        
        self.data_dir = data_dir
        self.mode = mode
        self.delta_t = delta_t
        self.initial_size = initial_size
        self.crop_size = crop_size
        self.repeat_sample = repeat_sample
        self.backend = backend
        
        assert backend in ['decord', 'torchvision', 'torchvision-videoreader'], f"backend must be one of ['decord', 'torchvision', 'torchvision-videoreader'], got {backend}"
        if backend == 'torchvision-videoreader':
            torchvision.set_video_backend('video_reader')
        else:
            torchvision.set_video_backend('pyav')

        video_dir = os.path.join(data_dir, mode)
        video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir)]
        if meta_info_file is not None:
            meta_info = np.load(meta_info_file, allow_pickle=True).item()
            self.video_paths = []
            self.video_metadata = []
            for p in video_paths:
                info = meta_info.get(p)
                if info is None:
                    continue
                length = info.get('length', 0)
                if length <= (delta_t[1] + 1):
                    continue
                self.video_paths.append(p)
                self.video_metadata.append(info)
        else:
            self.video_paths = video_paths
            self.video_metadata = [None for _ in video_paths]
        self._dataset_len = len(self.video_paths)

        self.transform = transformsv2.Compose(
            [
                transformsv2.RandomResize(min_size=initial_size[0], max_size=initial_size[1]),
                transformsv2.RandomCrop(crop_size),
                transformsv2.ToImage(),
                transformsv2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.aug = transformsv2.Compose(
            [
                transformsv2.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transformsv2.RandomGrayscale(0.2),
                transformsv2.RandomApply([transforms.GaussianBlur(51)], p=0.5),
                transformsv2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            ]
        )
       
    def __len__(self):
        return self._dataset_len
    
    def __getitem__(self, idx):
        try:
            worker_info = torch.utils.data.get_worker_info()
            cpuid = 0 if worker_info == None else int(get_rank() * worker_info.num_workers + (worker_info.id))
            if self.backend == 'decord':
                vr = VideoReader(self.video_paths[idx], num_threads=0, ctx=cpu(cpuid))
                vr_len = len(vr)
            elif 'torchvision' in self.backend:
                vr = TVVideoReader(self.video_paths[idx], "video")
                # conda-base FFMPEG does not preserve rotations properly, must read manually
                if self.video_metadata[idx] is not None and 'rotation' in self.video_metadata[idx]:
                    vr_rotation = int(self.video_metadata[idx]['rotation'])
                else:
                    try:
                        vr_rotation = -int(ffprobe(self.video_paths[idx]).json['streams'][0]['side_data_list'][0].get('rotation', '0'))
                    except:
                        vr_rotation = 0
                vr_md = vr.get_metadata()['video']
                vr_len = int(vr_md['duration'][0] * vr_md['fps'][0]) - 1

            delta_t = np.random.randint(self.delta_t[0], self.delta_t[1]+1)
            i_s = np.random.randint(0, vr_len - delta_t, size=self.repeat_sample)
            i_s = np.array([index for i in i_s for index in [i, i+delta_t]])
            sort_indexes = np.argsort(i_s).astype(np.int32)
            unsort_indexes = np.argsort(sort_indexes).astype(np.int32)
            if self.backend == 'decord':
                imgs = vr.get_batch(list(i_s[sort_indexes])).asnumpy()[unsort_indexes]
            elif 'torchvision' in self.backend:
                res = []
                i_s_ = [x / vr_md['fps'][0] for x in i_s[sort_indexes]]
                for i_ in i_s_:
                    vr.seek(i_)
                    count = 0
                    while True:
                        try:
                            res.append(next(vr)['data'])
                            break
                        except StopIteration:
                            log.warning(f"StopIteration at {i_ + (count * 1/vr_md['fps'][0])} for {self.video_paths[idx]}")
                            count += 1
                            if count < 3:
                                vr.seek(i_ + count * 1/vr_md['fps'][0])
                            else:
                                log.warning(f"Failed to read frame for 3rd iteration, resorting to keyframe from {i_-1/vr_md['fps'][0]}")
                                vr.seek(i_-1/vr_md['fps'][0], keyframes_only=True)
                                decode_res = next(vr)
                                log.warn(f"Keyframe from {i_} read succesfully at {decode_res['pts']}")
                                res.append(decode_res['data'])
                                break
                imgs = torch.stack(res, axis=0)
                if vr_rotation != 0:
                    imgs = torch.rot90(imgs, k=-vr_rotation//90, dims=[2, 3])
                imgs = imgs.permute(0, 2, 3, 1).numpy()[unsort_indexes]
            del vr; gc.collect()

            ls = []
            for i in range(0, 2*self.repeat_sample, 2):
                img1 = Image.fromarray(imgs[i])
                img2 = Image.fromarray(imgs[i+1])
                img1, img2 = self.transform(img1, img2)
                aug_img1 = self.aug(img1)
                aug_img2 = self.aug(img2)
                res = [img1, img2, aug_img1, aug_img2]
                if len(ls) == 0:
                    [ls.append([]) for _ in range(len(res))]
                for j, x in enumerate(res):
                    ls[j].append(x)
            return tuple(torch.stack(x, dim=0) for x in ls)
        except Exception as e:
            log.error(f"Error in {self.video_paths[idx]}: {e}")
            return self.__getitem__(np.random.randint(0, len(self.video_paths)))
