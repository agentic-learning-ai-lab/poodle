
import logging
import math
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from spatial_correlation_sampler import SpatialCorrelationSampler

from src.models.registry import register_model
from src.utils.flow import flow_to_warp, mask_invalid, normalize_features, resize_flow, upsample_flow, warp, compute_occlusion_mask, compute_selfsup_mask
from src.utils.losses import reconstruction_loss, smoothness_loss
from src.models.backbone.pwc_backbone import PWCBackbone

log = logging.getLogger(__name__)

ESTIMATOR_MODULE_SETUPS = [
    ['flow', 'flow', 'flow', 'flow', 'flow'],
    ['none', 'flow', 'flow', 'flow', 'flow'],
    ['flow', 'flow', 'flow', 'none', 'none'],
    ['none', 'flow', 'flow', 'none', 'none'],
]

def convert_str_to_estimator_modules(s):
    groups = s.split("-")
    groups = [(int(group.split("_")[0]), group.split("_")[1]) for group in groups]
    module_list = []
    for num, module in groups:
        module_list.extend([module] * num)
    return module_list

def _tf_init(m):
    if isinstance(m, (torch.nn.ConvTranspose2d, torch.nn.Conv2d, torch.nn.Linear)):
        if m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class UFlow(nn.Module):
    def __init__(self,
                 backbone: str,
                 inference_res: List[int],
                 # flow estimator
                 estimator_modules='5_flow', # 'flow' or 'none'
                 flow_predictor_width='base', # 'base' or 'scaled'
                 downproj_feat='scaled', # 'scaled' or 'constant' or 'inverted' or 'nonlineartop'
                 pre_corr_norm='layernorm', # norm or layernorm
                 inter_conv_norm=True,
                 leaky_relu_alpha=0.1,
                 dropout_p=0.1,
                 accumulate_flow=True,
                 corr_patch_size=9,
                 context_dim=32,
                 refinement_module=False,
                 refinement_module_dims=[(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)],
                 # flow-self-sup,
                 selfsup_params=None,
                 # other/loss arguments
                 alpha1=0.01,
                 alpha2=0.5,
                 smoothness_level='pred', # 'image' or 'pred'
                 smoothness_lambda=150,
                 smoothness_order=2,
    ):
        super().__init__()
        if isinstance(estimator_modules, str):
            estimator_modules = convert_str_to_estimator_modules(estimator_modules)
        if isinstance(refinement_module_dims, str):
            if refinement_module_dims == 'base':
                refinement_module_dims = [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]
            else: raise ValueError(f"Invalid refinement_module_dims {refinement_module_dims}")
        
        assert estimator_modules in ESTIMATOR_MODULE_SETUPS, f"Invalid estimator_modules {estimator_modules}"
        self.inference_res = inference_res
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.smoothness_level = smoothness_level
        self.smoothness_lambda = smoothness_lambda
        self.smoothness_order = smoothness_order
        
        self.last_flow_level = estimator_modules.index('flow') # 0
        self.first_flow_level = (len(estimator_modules) - 1) - estimator_modules[::-1].index('flow') # 4
        
        self.pool_last2 = False
        if backbone == 'pwc':
            self.encoder = PWCBackbone()
            downsample_ratio = [2, 2, 2, 2, 2]
            encoder_channels = [32, 32, 32, 32, 32]
            for i in range(self.first_flow_level+1, len(encoder_channels)):
                self.encoder.blocks[i] = nn.Identity()
        else: 
            raise NotImplementedError(f"Invalid backbone {backbone}")
            
        assert all([x % np.prod(downsample_ratio) == 0 for x in self.inference_res]), f"Invalid inference_res {self.inference_res}"
        assert len(estimator_modules) == len(downsample_ratio)
        self.downsample_ratio = downsample_ratio
        
        self.flow_predictor = FlowPredictor(
            backbone_feature_dims=encoder_channels,
            backbone_downsample_ratios=downsample_ratio,
            estimator_modules=estimator_modules,
            flow_predictor_width=flow_predictor_width,
            downproj_feat=downproj_feat,
            pre_corr_norm=pre_corr_norm,
            inter_conv_norm=inter_conv_norm,
            leaky_relu_alpha=leaky_relu_alpha,
            dropout_p=dropout_p,
            accumulate_flow=accumulate_flow,
            corr_patch_size=corr_patch_size,
            context_dim=context_dim,
            refinement_module=refinement_module,
            refinement_module_dims=refinement_module_dims
        )
        
        self.apply(_tf_init)
        
        # Self-supervision
        self.selfsup_params: dict = selfsup_params
        if selfsup_params:
            self.selfsup_aug_level0 = transforms.Compose([
                transforms.CenterCrop([x-y for x, y in zip(inference_res, self.selfsup_params.crop)]),
                transforms.Resize(inference_res)
            ])
            self.selfsup_loss_stride = np.prod(downsample_ratio[:self.last_flow_level+1])
            self.selfsup_loss_res = [x // self.selfsup_loss_stride for x in inference_res]
            self.selfsup_aug_level2 = transforms.Compose([
                transforms.CenterCrop([x-(y//self.selfsup_loss_stride) for x, y in zip(self.selfsup_loss_res, selfsup_params.crop)]),
                transforms.Resize(self.selfsup_loss_res),
            ])
    
    @torch.no_grad()
    def _update_momentum_encoder(self, m, source: Iterable[torch.nn.Module]=None, target: Iterable[torch.nn.Module]=None):
        raise NotImplementedError("Momentum encoder not supported")
    
    def resize_for_encoder(self, x):
        divisible_by_num = np.prod(self.downsample_ratio)
        if self.inference_res is not None:
            H, W = self.inference_res
        else:
            H = int(math.ceil(float(H) / divisible_by_num) * divisible_by_num)
            W = int(math.ceil(float(W) / divisible_by_num) * divisible_by_num)
        if (H, W) != x.shape[-2:]:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x
        
    def forward(self, 
                source, target,
                return_features=False, return_all_scales=False,
                # training mode settings
                return_loss=False, 
                source_unaug=None, target_unaug=None,
                gt_flow=None, gt_flow_mask=None,
                loss_weights=None,
                use_occ_masking=None, use_selfsup=None, use_ssl=None, momentum=None):
        assert not use_ssl and momentum is None, "use_ssl and momentum are not supported"
        source_orig, target_orig = source, target
        N, _, H_inp, W_inp = source.shape
        
        # resize so that input is power of 2 to fit encoder
        source, target = self.resize_for_encoder(source), self.resize_for_encoder(target)
        
        # set encoder inputs to [-1, 1]
        source_encoder = source * 2 - 1.
        target_encoder = target * 2 - 1.
        
        if not return_loss:
            feature_pyramid1 = self.encoder.forward_feature_pyramid(source_encoder)
            feature_pyramid2 = self.encoder.forward_feature_pyramid(target_encoder)
            
            if self.pool_last2:
                feature_pyramid1 = feature_pyramid1[:-2] + [F.avg_pool2d(feature_pyramid1[-2], 2), F.avg_pool2d(feature_pyramid1[-1], 4)]
                feature_pyramid2 = feature_pyramid2[:-2] + [F.avg_pool2d(feature_pyramid2[-2], 2), F.avg_pool2d(feature_pyramid2[-1], 4)]
            flow_pyramid = self.flow_predictor(feature_pyramid1, feature_pyramid2)
            
            if (H_inp, W_inp) != flow_pyramid[0].shape[-2:]:
                flow_pyramid[0] = resize_flow(flow_pyramid[0], H_inp, W_inp)
            
            ret = flow_pyramid if return_all_scales else flow_pyramid[0]
            if return_features: 
                return ret, feature_pyramid1, feature_pyramid2
            else:
                return ret
        else:
            assert loss_weights is not None
            assert use_occ_masking is not None
            
            # forward features and split into source and target
            sourcetarget_ = torch.cat([source_encoder, target_encoder], dim=0)
            features_sourcetarget = self.encoder.forward_feature_pyramid(sourcetarget_)
            
            if self.pool_last2:
                features_sourcetarget = features_sourcetarget[:-2] + [F.avg_pool2d(features_sourcetarget[-2], 2), F.avg_pool2d(features_sourcetarget[-1], 4)] 
            
            features_source, features_target = [x[:N] for x in features_sourcetarget], [x[N:] for x in features_sourcetarget]
            features_targetsource = [torch.cat((features_target[i], features_source[i]), dim=0) for i in range(len(features_source))]
            flow_pyramid = self.flow_predictor(features_sourcetarget, features_targetsource)
            
            # combine targetsource for reverse flows
            sourcetarget = torch.cat([source, target], dim=0)
            flow_pyramid_reverse = [x.reshape(2, N, *x.shape[1:]).flip(0).flatten(0, 1) for x in flow_pyramid]
            
            # -- compute loss -- 
            raw_loss_dict = {}
            metric_dict = {}

            for i, f in enumerate(flow_pyramid):
                metric_dict[f'fmean{i}'] = torch.linalg.norm(f, dim=1).mean()
            
            warp_pyramid = [flow_to_warp(flow) for flow in flow_pyramid]
            with torch.no_grad():
                forward_valid_mask = mask_invalid(warp_pyramid[0])
                if use_occ_masking:
                    occ_mask = compute_occlusion_mask(flow_pyramid[0], flow_pyramid_reverse[0], alpha1=self.alpha1, alpha2=self.alpha2)
                    mask = occ_mask * forward_valid_mask
                    metric_dict['occ_mask'] = occ_mask.mean()
                else:
                    mask = forward_valid_mask
                metric_dict['valid_mask'] = forward_valid_mask.mean()
                metric_dict['mask'] = mask.mean()
                mask = mask.detach()
            
            # image-only losses
            source_unaug = source_orig if source_unaug is None else self.resize_for_encoder(source_unaug)
            target_unaug = target_orig if target_unaug is None else self.resize_for_encoder(target_unaug)
            
            sourcetarget_unaug = torch.cat([source_unaug, target_unaug], dim=0)
            targetsource_unaug = torch.cat([target_unaug, source_unaug], dim=0)
            l1, l2, ssim = reconstruction_loss(sourcetarget_unaug, targetsource_unaug, flow_pyramid[0], mask)
            raw_loss_dict['recon_l1'] = l1
            raw_loss_dict['recon_l2'] = l2
            raw_loss_dict['recon_ssim'] = ssim
            
            # smoothness
            if loss_weights.get('smoothness', 0.) > 0.:
                if self.smoothness_level == 'image':
                    smoothness = smoothness_loss(sourcetarget, flow_pyramid[0], smoothness_lambda=self.smoothness_lambda, order=self.smoothness_order)
                elif self.smoothness_level == 'pred':
                    pred_level = self.flow_predictor.last_flow_level + 1
                    s_img = sourcetarget
                    for _ in range(pred_level):
                        s_img = F.interpolate(s_img, scale_factor=0.5, mode='bilinear', align_corners=False)
                    smoothness = smoothness_loss(s_img, flow_pyramid[pred_level], smoothness_lambda=self.smoothness_lambda, order=self.smoothness_order)
                raw_loss_dict['smoothness'] = smoothness
            
            if self.selfsup_params and loss_weights.get('selfsup', 0.) > 0. and use_selfsup: 
                # compute student flows
                with torch.set_grad_enabled(not self.selfsup_params.get('stop_student_encoder_gradient', False)):
                    selfsup_student_sourcetarget = self.selfsup_aug_level0(sourcetarget_)
                    s_features_sourcetarget = self.encoder.forward_feature_pyramid(selfsup_student_sourcetarget)
                s_features_source, s_features_target = [x[:N] for x in s_features_sourcetarget], [x[N:] for x in s_features_sourcetarget]
                student_flow = self.flow_predictor(s_features_sourcetarget, [
                    torch.cat((s_features_target[i], s_features_source[i]), dim=0) for i in range(len(s_features_source))
                ])[2]
                
                # get teacher modules
                t_encoder, t_flow_predictor = self.encoder, self.flow_predictor
                
                # compute teacher flows and masks (no gradients)
                with torch.no_grad():
                    if self.selfsup_params.get('teacher_unaug', False):
                        t_features_sourcetarget = t_encoder.forward_feature_pyramid(sourcetarget_unaug * 2 - 1.)
                        t_features_source, t_features_target = [x[:N] for x in t_features_sourcetarget], [x[N:] for x in t_features_sourcetarget]
                        teacher_flows = t_flow_predictor(t_features_sourcetarget, [
                            torch.cat((t_features_target[i], t_features_source[i]), dim=0) for i in range(len(t_features_source))
                        ])
                    else:
                        teacher_flows = [x.detach() for x in flow_pyramid]
                    crop_size = (self.selfsup_params.crop[0] // self.selfsup_loss_stride, self.selfsup_params.crop[1] // self.selfsup_loss_stride)
                    teacher_flow = self.selfsup_aug_level2(teacher_flows[2]) * torch.tensor([x/(x-y) for (x, y) in zip(self.selfsup_loss_res[::-1], crop_size)], device=source.device)[None, :, None, None]
                    student_mask = 1. - compute_selfsup_mask(student_flow, torch.cat((student_flow[N:], student_flow[:N]), dim=0),
                                                             self.selfsup_params.fb_sigma_student)
                    teacher_mask = compute_selfsup_mask(teacher_flows[2], torch.cat((teacher_flows[2][N:], teacher_flows[2][:N]), dim=0),
                                                        self.selfsup_params.fb_sigma_teacher)
                    teacher_mask = self.selfsup_aug_level2(teacher_mask) # apply augmentation so they line up
                    mask = (student_mask * teacher_mask).detach()
                    metric_dict['selfsup_mask'] = mask.mean()
                selfsup_loss = ((student_flow - teacher_flow)**2 + 1e-6)**0.5
                selfsup_loss = ((selfsup_loss * mask).sum(dim=(1,2,3)) / (mask.sum(dim=(1,2,3)) + 1e-6)).mean()
                raw_loss_dict['selfsup'] = selfsup_loss

            loss_dict = {k: v * loss_weights.get(k, 0.) for k, v in raw_loss_dict.items()}

            loss = sum(loss_dict.values())
            return loss, loss_dict, raw_loss_dict, metric_dict

class FlowPredictor(nn.Module):
    def __init__(self,
                 backbone_feature_dims, # input feature pyramid dimensions
                 backbone_downsample_ratios, # backbone downsample ratios
                 estimator_modules=['none', 'flow', 'flow', 'flow', 'flow'], # modules to use for each level
                 flow_predictor_width='base', # 'base' or 'scaled' or 'small' or 'inverted' or 'per-level'
                 downproj_feat='scaled', # 'scaled' or 'constant' or 'inverted' or 'nonlineartop'
                 pre_corr_norm='norm', # 'norm' or 'layernorm'
                 inter_conv_norm=True,
                 leaky_relu_alpha=0.1,
                 accumulate_flow=True,
                 dropout_p=0.1,
                 corr_patch_size=9,
                 context_dim=32,
                 refinement_module=True,
                 refinement_module_dims=[(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]):
        super().__init__()
        
        self.backbone_downsample_ratios = backbone_downsample_ratios
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_p = dropout_p
        self.accumulate_flow = accumulate_flow
        self.corr_patch_size = corr_patch_size
        self.context_dim = context_dim
        self.pre_corr_norm = pre_corr_norm
        self.downproj_feat = downproj_feat
        self.inter_conv_norm = inter_conv_norm
        
        self.dense_conv = True
        if flow_predictor_width == 'base':
            flow_module_dims = [(128, 128, 96, 64, 32)] * len(estimator_modules)
        elif flow_predictor_width == 'scaled':
            flow_module_dims = [(64, 32), (64, 32), (96, 64, 32), (128, 96, 64, 32), (128, 128, 96, 64, 32)]
        elif flow_predictor_width == 'small':
            flow_module_dims = [(64, 32)] * len(estimator_modules)
        elif flow_predictor_width == 'inverted':
            self.dense_conv = False
            flow_module_dims = [(512, 32)] * len(estimator_modules)
        elif flow_predictor_width == 'per-level':
            flow_module_dims = [[0] * 5, [32] * 5, [64] * 5, [96]*5, [128] * 5]
        elif flow_predictor_width == 'flat':
            flow_module_dims = [(32, 32, 32, 32, 32)] * len(estimator_modules)
        else:
            raise ValueError(f"Invalid flow_predictor_width {flow_predictor_width}")
        
        flow_module_dims = [None if module == 'none' else dims for module, dims in zip(estimator_modules, flow_module_dims)]
        
        if downproj_feat: 
            nonlinear_downproj = False
            if downproj_feat == 'constant':
                downproj_dims = [32] * len(estimator_modules)
            elif downproj_feat == 'scaled':
                downproj_dims = [64, 64, 96, 96, 128]
            elif isinstance(downproj_feat, int):
                downproj_dims = [downproj_feat] * len(estimator_modules)
            elif downproj_feat == 'inverted':
                downproj_dims = [32] * len(estimator_modules)
                nonlinear_downproj = True
            elif downproj_feat == 'nonlineartop':
                downproj_dims = [32, 32, 32, 32, 32]
            else:
                raise ValueError(f"Invalid downproj_feat {downproj_feat}")
        else:
            downproj_dims = None
        self.downproj_dims = downproj_dims
        
        assert pre_corr_norm in ['norm', 'layernorm', None]
        
        # build flow modules
        self.last_flow_level = estimator_modules.index('flow') # 0
        self.first_flow_level = (len(estimator_modules) - 1) - estimator_modules[::-1].index('flow') # 4
        
        if self.downproj_dims:
            self.downproj_layers = nn.ModuleList()
        flow_modules = [] 
        for i, (module_name, module_dims, inp_feature_dim, upsample_ratio)  in enumerate(zip(estimator_modules,
                                                                                         flow_module_dims,
                                                                                         backbone_feature_dims,
                                                                                         backbone_downsample_ratios)):
            if downproj_dims:
                if module_name == 'flow':
                    if downproj_feat == 'nonlineartop':
                        if i == self.first_flow_level: # if top layer
                            self.downproj_layers.append(nn.Sequential(
                                nn.Conv2d(inp_feature_dim, int(inp_feature_dim*1.5), kernel_size=1),
                                nn.BatchNorm2d(int(inp_feature_dim*1.5)),
                                nn.LeakyReLU(leaky_relu_alpha, inplace=True),
                                nn.Conv2d(int(inp_feature_dim*1.5), downproj_dims[i], kernel_size=1),
                            ))
                        else:
                            self.downproj_layers.append(nn.Conv2d(inp_feature_dim, downproj_dims[i], kernel_size=1))
                    else:
                        if nonlinear_downproj:
                            projdim_ = int(inp_feature_dim*1.5)
                            self.downproj_layers.append(nn.Sequential(
                                nn.Conv2d(inp_feature_dim, projdim_, kernel_size=1),
                                nn.BatchNorm2d(projdim_),
                                nn.LeakyReLU(leaky_relu_alpha, inplace=True),
                                nn.Conv2d(projdim_, downproj_dims[i], kernel_size=1),
                            ))
                        else:
                            self.downproj_layers.append(nn.Conv2d(inp_feature_dim, self.downproj_dims[i], kernel_size=1))
                    inp_feature_dim = self.downproj_dims[i]
                elif module_name == 'none': 
                    self.downproj_layers.append(nn.Identity())
            if module_name == 'flow':
                layers = []
                curr_dim = corr_patch_size ** 2 + inp_feature_dim + (2 if i < self.first_flow_level else 0) + (context_dim if i < self.first_flow_level else 0)
                for j, c in enumerate(module_dims):
                    optional_layer_norm = (LayerNorm(c, eps=1e-6, data_format="channels_first"),) if inter_conv_norm else ()
                    layers.append(nn.Sequential(
                        nn.Conv2d(curr_dim, c, 3, padding=1, stride=1),
                        *optional_layer_norm,
                        nn.LeakyReLU(leaky_relu_alpha, inplace=True,)
                    ))
                    
                    curr_dim = (curr_dim + c) if self.dense_conv else c
                layers.append(nn.Conv2d(module_dims[-1], 2, 3, padding=1, stride=1))
                flow_modules.append(nn.Sequential(*layers))
            if module_name == 'none':
                assert module_dims is None
                flow_modules.append(nn.Identity())
            
        self.flow_modules = nn.ModuleList(flow_modules)
            
        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=corr_patch_size, stride=1, padding=0, dilation_patch=1)
        self.corr_relu = nn.LeakyReLU(leaky_relu_alpha, inplace=True)
        
        if pre_corr_norm == 'layernorm':
            norm_layers = []
            for level, module in enumerate(estimator_modules):
                inp_feature_dim = backbone_feature_dims[level] if downproj_dims is None else downproj_dims[level]
                if module == 'flow':
                    norm_layers.append(LayerNorm(inp_feature_dim, eps=1e-6, data_format="channels_first"))
                else:
                    norm_layers.append(nn.Identity())
            self.norm_layers = nn.ModuleList(norm_layers)
        
        upsample_modules = []
        for level, upsample_ratio in enumerate(backbone_downsample_ratios):
            # build upsample module if before lowest flow level, and after or at first flow level
            if level > self.last_flow_level and level <= self.first_flow_level:
                inp_dim = context_dim if flow_module_dims[level] is None else flow_module_dims[level][-1]
                if upsample_ratio == 2:
                    upsample_modules.append(nn.ConvTranspose2d(inp_dim, context_dim, 4, padding=1, stride=2))
                elif upsample_ratio == 1:
                    upsample_modules.append(nn.Conv2d(inp_dim, context_dim, 3, padding=1, stride=1))
                else:
                    raise ValueError(f"Invalid upsample ratio {upsample_ratio}")
            else:
                upsample_modules.append(nn.Identity())
        self.upsample_modules = nn.ModuleList(upsample_modules)
            
        
        self.dropout_p = dropout_p
                
        if refinement_module:
            layers = []
            curr_dim = context_dim + 2
            for (c, d) in refinement_module_dims:
                layers.append(nn.Conv2d(curr_dim, c, 3, padding=d, stride=1, dilation=d))
                layers.append(nn.LeakyReLU(leaky_relu_alpha, inplace=True))
                curr_dim = c
            layers.append(nn.Conv2d(curr_dim, 2, 3, padding=1, stride=1))
            self.refinement_module = nn.Sequential(*layers)
        else:
            self.refinement_module = None
    
    def forward(self, feature_pyramid1, feature_pyramid2):
        context = None
        flow = None
        flow_up = None
        context_up = None
        flows = []
        
        for level, (feature1, feature2) in reversed(
            list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[self.last_flow_level:self.first_flow_level+1]):
            if self.downproj_dims:
                feature1 = self.downproj_layers[level](feature1)
                feature2 = self.downproj_layers[level](feature2)
            
            if self.pre_corr_norm == 'layernorm':
                feature1 = self.norm_layers[level](feature1)
                feature2 = self.norm_layers[level](feature2)
            
            if flow_up is None:
                warped2 = feature2
            else:
                warp_ = flow_to_warp(flow_up)
                warped2 = warp(feature2, warp_, convert_flow_to_coords=False).contiguous()
            
            # compute cost volume between feature1 and warped2
            if self.pre_corr_norm == 'norm':
                feature1_norm, warped2_norm = normalize_features(
                    [feature1, warped2],
                    normalize=True, center=True,
                    across_channels=True, across_features=True
                )
                cost_volume = self.corr(feature1_norm.type(torch.float32), warped2_norm.type(torch.float32)).flatten(1, 2)/feature1_norm.shape[1]
            else:
                cost_volume = self.corr(feature1.type(torch.float32), warped2.type(torch.float32)).flatten(1, 2)/feature1.shape[1]
            cost_volume = self.corr_relu(cost_volume.type(feature1.dtype))
            
            # Compute context and flow from previous flow, cost volume and features1
            if flow_up is None:
                x_in = torch.cat([cost_volume, feature1], dim=1)
            else:
                if context_up is None:
                    x_in = torch.cat([flow_up, cost_volume, feature1], dim=1)
                else:
                    x_in = torch.cat([context_up, flow_up, cost_volume, feature1], dim=1)
            x_out = None
            
            for i, layer in enumerate(self.flow_modules[level][:-1]):
                x_out = layer(x_in)
                x_in = torch.cat([x_in, x_out], dim=1) if self.dense_conv else x_out
            context = x_out
            flow = self.flow_modules[level][-1](context)
            
            # dropout
            if self.dropout_p > 0. and self.training:
                if self.accumulate_flow or (not (level == self.last_flow_level)):
                    rand_dropout = (torch.rand((flow.shape[0],), device=flow.device) > self.dropout_p)[:, None, None, None]
                    flow = flow * rand_dropout.type(flow.dtype)
                    context = context * rand_dropout.type(context.dtype)
            
            # residual connection
            if flow_up is not None and self.accumulate_flow:
                flow += flow_up
            
            # upsample flow and context
            if self.backbone_downsample_ratios[level] == 2:
                flow_up = upsample_flow(flow, 2.)
            else: 
                flow_up = flow
            context_up = self.upsample_modules[level](context)
            
            flows.insert(0, flow)
        
        if self.refinement_module is not None:
            refinement = self.refinement_module(torch.cat([context, flow], dim=1))
            if self.dropout_p > 0. and self.training:
                rand_dropout = (torch.rand((flow.shape[0],), device=flow.device) > self.dropout_p)[:, None, None, None]
                refinement = refinement * rand_dropout.type(refinement.dtype)
            flows[0] += refinement
        
        for _ in range(self.last_flow_level+1):
            flows.insert(0, upsample_flow(flows[0], 2.))
        
        return flows

@register_model
def uflow(**kwargs):
    return UFlow(**kwargs)    
