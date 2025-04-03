from copy import deepcopy
import math
from typing import List

import kornia.augmentation as KA
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models.resnet import Bottleneck

from src.models.backbone.resnet import resnet50, resnet50_stride8
from src.models.raft.raft import RAFT
from src.models.registry import register_model
from src.utils.flow import flow_to_warp, mask_invalid, warp, compute_occlusion_mask
from src.utils.bbox import resize_crop_to_bounding_boxes

MAX_TRIES_PER_CROP = 5


class Poodle(nn.Module):
    def __init__(self,
                 backbone_type: str,
                 flow_model_checkpoint: str,
                 pretrained_backbone: bool=False,
                 affine_aug_params: dict=None,
                 mlp_hidden_dim: int=4096,
                 proj_dim: int=256,
                 # occlusion
                 occlusion_masking: bool=True,
                 alpha1: float=1.0,
                 alpha2: float=1.0,
                 # local crop
                 subcrop_num_pair: int=6,
                 subcrop_num_same: int=0,
                 subcrop_scale=[0.05, 0.3],
                 subcrop_ratio=[3. / 4., 4. / 3.],
                 subcrop_size=[192, 384],
                 subcrop_jitter: float=0.,
                 # spatial decoder
                 decoder_dims: List[int]=[2048, 2048],
                 decoder_hidden_dim_factor: float=0.5,
                 ):
    
        super().__init__()
        self.backbone_type = backbone_type

        self.affine_aug_params = affine_aug_params
        if self.affine_aug_params is not None:
            self.affine_aug = KA.RandomAffine(**self.affine_aug_params, p=1.0)
            # assume horizontal flip
            self.flip_aug_online = KA.RandomHorizontalFlip()
            self.flip_aug_teacher = KA.RandomHorizontalFlip()
        self.use_affine_aug = self.affine_aug_params is not None 

        # occlusion attributes
        self.occlusion_masking = occlusion_masking
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        # backbone
        if self.backbone_type == 'resnet50-stride8':
            self.backbone = resnet50_stride8(pretrained=pretrained_backbone)
            self.backbone_dims = [256, 512, 1024, 2048]
        elif self.backbone_type == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained_backbone)
            self.backbone_dims = [256, 512, 1024, 2048]
        else:
            raise ValueError(f'Invalid backbone type: {backbone_type}')
        self.teacher_backbone = deepcopy(self.backbone)
        self.num_backbone_layers = len(self.backbone_dims)

        # decoder
        self.decoder_dims = decoder_dims
        self.decoder_blocks = nn.ModuleList()
        self.decoder_laterals = nn.ModuleList()
        decoder_dims = [self.backbone_dims[-1], *decoder_dims]
        num_dims = len(decoder_dims) - 1
        for i in range(num_dims):
            downsample_dim = decoder_dims[i+1]
            if decoder_dims[i] != downsample_dim:
                downsample = nn.Sequential(
                    nn.Conv2d(decoder_dims[i], downsample_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(downsample_dim),
                )
            else:
                downsample = None
            block = Bottleneck(decoder_dims[i], downsample_dim // 4, downsample=downsample, base_width=int(64 * decoder_hidden_dim_factor))
            self.decoder_blocks.append(block)
            if self.backbone_dims[-i - 2] != downsample_dim:
                lateral = nn.Conv2d(self.backbone_dims[-i - 2], downsample_dim, kernel_size=1)
            else:
                lateral = nn.Identity()
            self.decoder_laterals.append(lateral)
        self.teacher_decoder_blocks = deepcopy(self.decoder_blocks)
        self.teacher_decoder_laterals = deepcopy(self.decoder_laterals)
        self.encoder_stages = list(range(len(self.backbone_dims) - len(self.decoder_laterals) - 1, len(self.backbone_dims)))
        
        # local crop attributes
        self.subcrop_num_pair = subcrop_num_pair
        self.subcrop_num_same = subcrop_num_same
        self.num_subcrops = subcrop_num_pair + subcrop_num_same
        self.subcrop_scale = tuple(subcrop_scale)
        self.subcrop_ratio = tuple(subcrop_ratio)
        self.subcrop_size = tuple(subcrop_size)
        self.subcrop_jitter = subcrop_jitter

        # flow model
        self.flow_model = RAFT()
        for param in self.flow_model.parameters():
            param.requires_grad = False
        self.flow_model.eval()
        loaded_flow_model_state_dict = torch.load(flow_model_checkpoint)
        flow_model_state_dict = {}
        for key in loaded_flow_model_state_dict.keys():
            flow_model_state_dict[key.replace('module.', '')] = loaded_flow_model_state_dict[key]
        self.flow_model.load_state_dict(flow_model_state_dict)

        self.mlp_hidden_dim = mlp_hidden_dim
        self.proj_dim = proj_dim

        # dense objective projector / predictor
        self.dense_projector = nn.Sequential(
            nn.Conv2d(decoder_dims[-1], self.mlp_hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mlp_hidden_dim, self.proj_dim, kernel_size=1)
        )
        self.dense_predictor = nn.Sequential(
                nn.Conv2d(self.proj_dim, self.mlp_hidden_dim, kernel_size=1),
                nn.BatchNorm2d(self.mlp_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mlp_hidden_dim, self.proj_dim, kernel_size=1)
        )
        self.teacher_dense_projector = deepcopy(self.dense_projector)

        # pooled objective projector / predictor
        self.pooled_projector = nn.Sequential(
            nn.Conv2d(self.backbone_dims[-1], self.mlp_hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mlp_hidden_dim, self.proj_dim, kernel_size=1)
        )
        self.pooled_predictor = nn.Sequential(
            nn.Conv2d(self.proj_dim, self.mlp_hidden_dim, kernel_size=1),
            nn.BatchNorm2d(self.mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mlp_hidden_dim, self.proj_dim, kernel_size=1)
        )
        self.teacher_pooled_projector = deepcopy(self.pooled_projector)

        for module in self.teacher_modules:
            for param in module.parameters():
                param.requires_grad = False

    @property
    def online_modules(self):
        return [self.backbone, self.dense_projector, self.pooled_projector,
                *self.decoder_blocks, *self.decoder_laterals]
    
    @property
    def teacher_modules(self):
        return [self.teacher_backbone, self.teacher_dense_projector, self.teacher_pooled_projector,
                *self.teacher_decoder_blocks, *self.teacher_decoder_laterals]

    @torch.no_grad()
    def _update_teacher(self, momentum: float):
        for module_base, module_momentum in zip(self.online_modules, self.teacher_modules):
            for param_base, param_momentum in zip(module_base.parameters(), module_momentum.parameters()):
                param_momentum.data = param_momentum.data * momentum + param_base.data * (1.0 - momentum)
    
    def apply_affine_aug(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            N, C, H, W = x.shape
            affine_mask = x.new_ones((N, 1, H, W))
            affine_x = self.affine_aug(x)
            affine_mask = self.affine_aug(affine_mask, self.affine_aug._params)
            affine_params = deepcopy(self.affine_aug._params)
            affine_mask = self.affine_aug.inverse(affine_mask, affine_params)
            affine_mask = (affine_mask - 1.0).abs() < 1e-3
        return affine_x, affine_mask, affine_params

    def forward(self, source, target,
                source_unaug, target_unaug,
                momentum):
        self._update_teacher(momentum)
        N, _, H, W = source.shape
        img_size = source.shape[-2:]
        metric_dict = {}

        with torch.no_grad():
            # compute flow between unaugmented and unaffined images
            forward_unaug = torch.cat([source_unaug, target_unaug], dim=0)
            backward_unaug = torch.cat([target_unaug, source_unaug], dim=0)
            forward_flow = self.flow_model(forward_unaug, backward_unaug, iters=24, test_mode=True)[1] # 2N, 2, H, W
            del forward_unaug, backward_unaug

            x = torch.cat([source, target], dim=0)
            teacher_x = torch.cat([target, source], dim=0)

            crop_search_resolution = round(math.sqrt(sum(self.subcrop_scale) / 2) * min(H, W))
            crop_search_grid_size = (H // crop_search_resolution, W // crop_search_resolution)
            crop_search_grid = torch.cartesian_prod(torch.arange(crop_search_grid_size[0] + 1, device=source.device),
                                                    torch.arange(crop_search_grid_size[1] + 1, device=source.device))
            online_crop_params = []
            warped_midpoints = []
            online_crop_scales = []
            use_pair = []
            for img_idx in range(2 * N):
                online_crop_params_ = []
                warped_midpoints_ = []
                online_crop_scales_ = []
                use_pair_ = []
                    
                use_replacement = self.num_subcrops > crop_search_grid.shape[0]
                sampled_crop_search_indices = torch.multinomial(torch.ones(crop_search_grid.shape[0]), num_samples=self.num_subcrops, replacement=use_replacement)
                for idx, (grid_i, grid_j) in enumerate(crop_search_grid[sampled_crop_search_indices]):
                    sampled_crop_scale = (self.subcrop_scale[1] - self.subcrop_scale[0]) * torch.rand(1)[0] + self.subcrop_scale[0]
                    _, _, h, w = T.RandomResizedCrop.get_params(teacher_x, scale=(sampled_crop_scale, sampled_crop_scale), ratio=self.subcrop_ratio)
 
                    # try to find crop in paired image
                    found_pair = False
                    if idx < self.subcrop_num_pair:
                        for _ in range(MAX_TRIES_PER_CROP):
                            i = max(torch.randint(grid_i * crop_search_resolution, min((grid_i + 1) * crop_search_resolution, H), (1,))[0], h // 2)
                            j = max(torch.randint(grid_j * crop_search_resolution, min((grid_j + 1) * crop_search_resolution, W), (1,))[0], w // 2)
                            midpoint = torch.tensor([i, j], device=source.device)
                            jitter = torch.tensor(img_size, device=source.device) * (2 * torch.rand(2, device=source.device) - 1) * self.subcrop_jitter # 2, (dy, dx)
                            crop_flow = forward_flow[img_idx, :, midpoint[0], midpoint[1]].flip(dims=(0,)) # dy, dx
                            warped_midpoint = torch.round(midpoint + crop_flow + jitter).to(torch.long) # (i, j) + (dy, dx)
                            if (warped_midpoint[0] >= 0) and (warped_midpoint[0] < H) and (warped_midpoint[1] >= 0) and (warped_midpoint[1] < W):
                                online_crop_params_.append([i - h // 2, j - w // 2, h, w])
                                warped_midpoints_.append(warped_midpoint)
                                online_crop_scales_.append(sampled_crop_scale)
                                use_pair_.append(True)
                                found_pair = True
                                break

                    # default to finding crop in same image
                    if not found_pair:
                        i = max(torch.randint(grid_i * crop_search_resolution, min((grid_i + 1) * crop_search_resolution, H), (1,))[0], h // 2)
                        j = max(torch.randint(grid_j * crop_search_resolution, min((grid_j + 1) * crop_search_resolution, W), (1,))[0], w // 2)
                        midpoint = torch.tensor([i, j], device=source.device)
                        warped_midpoint = midpoint.clone()
                        online_crop_params_.append([i - h // 2, j - w // 2, h, w])
                        warped_midpoints_.append(warped_midpoint)
                        online_crop_scales_.append(sampled_crop_scale)
                        use_pair_.append(False)

                online_crop_params.extend(online_crop_params_)
                warped_midpoints.extend(warped_midpoints_)
                online_crop_scales.extend(online_crop_scales_)
                use_pair.extend(use_pair_)

            online_crop_params = torch.tensor(online_crop_params, device=source.device) # 2N*K, 4
            warped_midpoints = torch.stack(warped_midpoints, dim=0) # 2N, 2
            teacher_crop_params = torch.tensor([T.RandomResizedCrop.get_params(teacher_x, scale=(online_crop_scales[i], self.subcrop_scale[1]), ratio=self.subcrop_ratio) for i in range(2*N*self.num_subcrops)], device=source.device) # i, j, h, w
                
            online_crop_params[:, 2:] = online_crop_params[:, :2] + online_crop_params[:, 2:] # y1, x1, y2, x2
            online_crop_params = online_crop_params.reshape(2*N, self.num_subcrops, 4) # N, K, 4

            teacher_crop_params[:, :2] = torch.round(warped_midpoints - (teacher_crop_params[:, 2:]//2)) # y1, x1, y2, x2
            teacher_crop_params[:, 2:] = teacher_crop_params[:, :2] + teacher_crop_params[:, 2:]
            teacher_crop_params[:, 0] = torch.clamp(teacher_crop_params[:, 0], 0, H)
            teacher_crop_params[:, 1] = torch.clamp(teacher_crop_params[:, 1], 0, W)
            teacher_crop_params[:, 2] = torch.clamp(teacher_crop_params[:, 2], 0, H)
            teacher_crop_params[:, 3] = torch.clamp(teacher_crop_params[:, 3], 0, W)
            teacher_crop_params = teacher_crop_params.reshape(2*N, self.num_subcrops, 4) # N, K, 4

            online_crop_params = online_crop_params[:, :, [1, 0, 3, 2]] # x1, y1, x2, y2
            teacher_crop_params = teacher_crop_params[:, :, [1, 0, 3, 2]]
            use_pair = torch.tensor(use_pair, device=source.device).to(torch.long).reshape(2*N, self.num_subcrops) # N, K
            
            # take subcrops, encode, and project
            teacher_pooled_z = resize_crop_to_bounding_boxes(torch.stack([x, teacher_x], dim=1), teacher_crop_params, self.subcrop_size, use_pair).flatten(0, 1) # 2N*K, 3, *self.subcrop_size
            teacher_pooled_z = self.teacher_backbone.forward_features(teacher_pooled_z, pool=True)
            teacher_pooled_z = self.teacher_pooled_projector(teacher_pooled_z) # 2N*K C, 1, 1
            teacher_pooled_z = F.normalize(teacher_pooled_z.flatten(1), p=2, dim=1) # 2NK, C

            # apply affine augmentation to teacher inputs
            if self.use_affine_aug:
                teacher_x, teacher_affine_mask, teacher_affine_param = self.apply_affine_aug(teacher_x)
                teacher_x = self.flip_aug_teacher(teacher_x)

            # encode teacher inputs
            teacher_features = self.teacher_backbone.forward_feature_pyramid(teacher_x, stages=self.encoder_stages)

            # teacher spatial decoder
            teacher_z = teacher_features[-1]
            for i in range(len(self.teacher_decoder_laterals)):
                teacher_z = self.teacher_decoder_blocks[i](teacher_z)
                teacher_z = F.interpolate(teacher_z, scale_factor=2, mode='bilinear', align_corners=False)
                teacher_z = teacher_z + self.teacher_decoder_laterals[i](teacher_features[-i - 2])
            for i in range(len(self.teacher_decoder_laterals), len(self.teacher_decoder_blocks)):
                teacher_z = self.teacher_decoder_blocks[i](teacher_z)

            teacher_z = self.teacher_dense_projector(teacher_z)
            # undo flip augmentation
            if self.use_affine_aug:
                teacher_z = self.flip_aug_teacher.inverse(teacher_z)
            # upsample features to img resolution
            teacher_z = F.interpolate(teacher_z, size=img_size, mode='bilinear', align_corners=False)
            # warp features to original (unaffined) frame
            teacher_z = warp(teacher_z, forward_flow, convert_flow_to_coords=True, grid_sample=True, mode='bilinear')
            # undo affine augmentations
            if self.use_affine_aug:
                teacher_z = self.affine_aug.inverse(teacher_z, teacher_affine_param)
            # L2 unit-normalize features
            teacher_z = F.normalize(teacher_z, p=2, dim=1)

            # compute masks
            forward_warp_ = flow_to_warp(forward_flow, for_grid_sample=True)
            forward_warp_mask = mask_invalid(forward_warp_, for_grid_sample=True) # N, 1, H, W
            transform_mask = forward_warp_mask
            if self.use_affine_aug:
                transform_mask = transform_mask * teacher_affine_mask.float()
                del teacher_affine_mask
            if self.occlusion_masking:
                # compute occlusion mask with image-resolution flow, then resize to feature resolution (if applicable)
                backward_flow = forward_flow.reshape(2, N, *forward_flow.shape[1:]).flip(0).flatten(0, 1)
                occlusion_mask = compute_occlusion_mask(forward_flow, backward_flow, alpha1=self.alpha1, alpha2=self.alpha2)
                mask = transform_mask * occlusion_mask
                del backward_flow, occlusion_mask
            else:
                mask = transform_mask    
            metric_dict['mask'] = mask.mean()

        # apply affine augmentation to online inputs
        if self.use_affine_aug:
            affine_x, online_affine_mask, online_affine_params = self.apply_affine_aug(x)
            affine_x = self.flip_aug_online(affine_x)
        else:
            affine_x = x

        # encode online inputs
        online_features = self.backbone.forward_feature_pyramid(affine_x, stages=self.encoder_stages)
        
        # online spatial decoder
        online_z = online_features[-1]
        for i in range(len(self.decoder_laterals)):
            online_z = self.decoder_blocks[i](online_z)
            online_z = F.interpolate(online_z, scale_factor=2, mode='bilinear', align_corners=False)
            online_z = online_z + self.decoder_laterals[i](online_features[-i - 2])
        for i in range(len(self.decoder_laterals), len(self.decoder_blocks)):
            online_z = self.decoder_blocks[i](online_z)

        # project and predict features
        online_z = self.dense_predictor(self.dense_projector(online_z))
        # undo flip augmentation
        if self.use_affine_aug:
            online_z = self.flip_aug_online.inverse(online_z)
        # upsample features to img resolution
        online_z = F.interpolate(online_z, size=img_size, mode='bilinear', align_corners=False)
        # undo affine augmentations
        if self.use_affine_aug:
            online_z = self.affine_aug.inverse(online_z, online_affine_params)
        # L2 unit-normalize features
        online_z = F.normalize(online_z, p=2, dim=1)
        
        with torch.no_grad():
            if self.use_affine_aug:
                mask = mask * online_affine_mask.float()
                del online_affine_mask

        # masked squared L2 distance loss, keep batch dimension for potential per-pixel weighting after
        dense_loss = 0.
        squared_z_diff = (online_z - teacher_z).pow(2).sum(dim=1, keepdim=True) * mask # 2N, 1, H, W
        dense_loss = ((squared_z_diff).sum(dim=(1,2,3)) / (mask.sum(dim=(1,2,3)) + 1e-7)).mean()
                
        # take subcrops, encode, project, and predict
        online_pooled_z = resize_crop_to_bounding_boxes(x, online_crop_params, self.subcrop_size).flatten(0, 1) # 2N*K, 3, *self.subcrop_size
        online_pooled_z = self.backbone.forward_features(online_pooled_z, pool=True)
        online_pooled_z = self.pooled_predictor(self.pooled_projector(online_pooled_z)) # 2N*K, C, 1, 1
        online_pooled_z = F.normalize(online_pooled_z.flatten(1), p=2, dim=1) # 2NK, C
        pooled_loss = (online_pooled_z - teacher_pooled_z).pow(2).sum(dim=1).mean()
        return {'dense': dense_loss, 'pooled': pooled_loss}, metric_dict

@register_model
def poodle(**kwargs):
    return Poodle(**kwargs)
