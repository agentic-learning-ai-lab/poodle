import copy

import kornia.augmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from timm.models.resnet import Bottleneck, ResNet

from src.models.raft.raft import RAFT
from src.models.registry import register_model


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True, mode=mode)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def compute_occlusion_mask(flow12, flow21, thresh=1.0):
    coords0 = coords_grid(flow12.shape[0], flow12.shape[2], flow12.shape[3], flow12.device)
    coords1 = coords0 + flow12
    coords2 = coords1 + bilinear_sampler(flow21, coords1.permute(0, 2, 3, 1))

    err = (coords0 - coords2).norm(dim=1)
    occ = err > thresh

    return occ

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    grid_x, grid_y = torch.meshgrid(torch.arange(0, H, device=x.device), torch.arange(0, W, device=x.device), indexing="ij")
    grid = torch.stack([grid_y, grid_x]).unsqueeze(0)

    vgrid = grid + flo
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    # mask = torch.ones((B, 1, H, W), device=x.device)
    # mask = F.grid_sample(mask, vgrid)

    # mask[mask < 0.999] = 0
    # mask[mask > 0] = 1
    mask = None

    return output, mask

class Flowe(nn.Module):
    def __init__(self,
                 flow_model_checkpoint: str,
                 occlusion_threshold: float):
        super().__init__()
        self.encoder = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=0, in_chans=3, output_stride=8, zero_init_last_bn=False)

        self.projector = nn.Sequential(
            nn.Conv2d(2048, 4096, 1, bias=False),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 256, 1),
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(256, 4096, 1, bias=False),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Conv2d(4096, 256, 1),
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)

        self.flownet = RAFT()

        def load_state_dict(path):
            state_dict = torch.load(path, map_location="cpu")
            for k in list(state_dict.keys()):
                state_dict[k[len("module.") :]] = state_dict[k]
                del state_dict[k]
            return state_dict

        state_dict = load_state_dict(flow_model_checkpoint)
        self.flownet.load_state_dict(state_dict)

        for m in (self.target_encoder, self.target_projector, self.flownet):
            for p in m.parameters():
                p.requires_grad = False

        self.data_aug = kornia.augmentation.RandomAffine(degrees=(-10, 10), scale=(0.9, 1.1), p=1.0)
        self.flip_aug_student = kornia.augmentation.RandomHorizontalFlip()
        self.flip_aug_teacher = kornia.augmentation.RandomHorizontalFlip()

        self.occlusion_threshold = occlusion_threshold

    @torch.no_grad()
    def _update_ema(self, momentum: float):
        for module_base, module_momentum in zip([self.encoder, self.projector], [self.target_encoder, self.target_projector]):
            for param_base, param_momentum in zip(module_base.parameters(), module_momentum.parameters()):
                param_momentum.data = param_momentum.data * momentum + param_base.data * (1.0 - momentum)

    def forward(self, x1, x2, v1, v2, momentum):
        self._update_ema(momentum)
        B, _, H, W = x1.shape

        # compute flow and flow mask
        with torch.no_grad():
            with torch.cuda.amp.autocast(False):
                _, flow = self.flownet(torch.cat([x1, x2]).float(), torch.cat([x2, x1]).float(), iters=30, test_mode=True)
                flow_2to1, flow_1to2 = flow.chunk(2, dim=0)
                flow_mask = ~compute_occlusion_mask(flow, torch.cat([flow_1to2, flow_2to1]), thresh=self.occlusion_threshold).unsqueeze(1)

        def data_aug(v):
            with torch.cuda.amp.autocast(False):
                affine_mask = v.new_ones((B * 2, 1, H, W))
                aug_v = self.data_aug(v)
                affine_mask = self.data_aug(affine_mask, self.data_aug._params)
                aug_param = copy.deepcopy(self.data_aug._params)
                affine_mask = self.data_aug.inverse(affine_mask, aug_param)
                affine_mask = (affine_mask - 1.0).abs() < 1e-3

            return aug_v, affine_mask, aug_param

        aug_student, affine_mask_student, aug_param_student = data_aug(torch.cat([v1, v2]))
        aug_teacher, affine_mask_teacher, aug_param_teacher = data_aug(torch.cat([v2, v1]))

        aug_student = self.flip_aug_student(aug_student)

        h_student = self.encoder.forward_features(aug_student)
        z_student = self.projector(h_student)
        p_student = self.predictor(z_student)
        p_student = self.flip_aug_student.inverse(p_student)

        with torch.cuda.amp.autocast(False):
            p_student = F.interpolate(p_student.float(), scale_factor=8, mode="bilinear", align_corners=False)

        # inverse affine
        p_student = self.data_aug.inverse(p_student, aug_param_student)

        with torch.no_grad():
            aug_teacher = self.flip_aug_teacher(aug_teacher)
            h_teacher = self.target_encoder.forward_features(aug_teacher)
            z_teacher = self.target_projector(h_teacher)
            z_teacher = self.flip_aug_teacher.inverse(z_teacher)

            # inverse flow
            with torch.cuda.amp.autocast(enabled=False):
                z_teacher = F.interpolate(z_teacher.float(), scale_factor=8, mode="bilinear", align_corners=False)
                z_teacher, _ = warp(z_teacher, flow.float())

            # inverse affine
            z_teacher = self.data_aug.inverse(z_teacher, aug_param_teacher)

            flow_mask = flow_mask & affine_mask_teacher
            flow_mask = flow_mask & affine_mask_student

        loss = self.loss(p_student, z_teacher.detach(), flow_mask.squeeze(1))

        return {'ssl': loss}, {}

    def loss(self, x1, x2, mask):
        loss = 2 - 2 * (F.normalize(x1, p=2, dim=1) * F.normalize(x2, p=2, dim=1)).sum(dim=1)
        return ((loss * mask).flatten(1, 2).sum(dim=-1) / (mask.flatten(1, 2).sum(dim=-1) + 1e-6)).mean()

@register_model
def flowe(**kwargs):
    return Flowe(**kwargs)
