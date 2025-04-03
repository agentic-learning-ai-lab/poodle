import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.raft.update import BasicUpdateBlock, SmallUpdateBlock
from src.models.raft.extractor import BasicEncoder, SmallEncoder
from src.models.raft.corr import CorrBlock, AlternateCorrBlock
from src.models.raft.utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self,
                 small=False,
                 dropout=0,
                 alternate_corr=False,
                 mixed_precision=False):
        super(RAFT, self).__init__()
        self.args = types.SimpleNamespace()

        if small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            self.args.corr_levels = 4
            self.args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            self.args.corr_levels = 4
            self.args.corr_radius = 4

        self.args.dropout = dropout
        self.args.alternate_corr = alternate_corr
        self.args.mixed_precision = mixed_precision

        # feature network, context network, and update block
        if small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=self.args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.args.dropout)        
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=self.args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device, dtype=img.dtype)
        coords1 = coords_grid(N, H//8, W//8, device=img.device, dtype=img.dtype)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0
        image1 = 2 * image1 - 1.0
        image2 = 2 * image2 - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])        
        
        with autocast(enabled=False):
            if self.args.alternate_corr:
                corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            else:
                corr_fn = CorrBlock(fmap1.type(torch.float32), fmap2.type(torch.float32), radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            with autocast(enabled=False):
                corr = corr_fn(coords1.type(torch.float32)).type(coords1.dtype) # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions

if __name__ == '__main__':
    import time

    flow_model_checkpoint = '/scratch/ch3451/models/raft/raft-sintel.pth'
    loaded_flow_model_state_dict = torch.load(flow_model_checkpoint)
    flow_model_state_dict = {}
    for key in loaded_flow_model_state_dict.keys():
        flow_model_state_dict[key.replace('module.', '')] = loaded_flow_model_state_dict[key]

    x1 = torch.rand(8, 3, 512, 1024).cuda()
    x2 = torch.rand(8, 3, 512, 1024).cuda()

    print('alternate_corr=False {}'.format('-' * 50))
    model = RAFT(alternate_corr=False)
    model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.load_state_dict(flow_model_state_dict)

    s = time.time()
    with torch.no_grad():
        flow = model(x1, x2, iters=24, test_mode=True)[1]
    e = time.time()
    print('flow computation time: {}'.format(e - s))
    print('gpu memory: {}'.format(torch.cuda.max_memory_allocated() / (1024. ** 3)))
    torch.cuda.reset_peak_memory_stats()
    del model

    print('alternate_corr=True {}'.format('-' * 50))
    model2 = RAFT(alternate_corr=True)
    model2 = model2.cuda()
    for param in model2.parameters():
        param.requires_grad = False
    model2.eval()
    model2.load_state_dict(flow_model_state_dict)

    s = time.time()
    with torch.no_grad():
        flow2 = model2(x1, x2, iters=24, test_mode=True)[1]
    e = time.time()
    print('flow computation time: {}'.format(e - s))
    print('gpu memory: {}'.format(torch.cuda.max_memory_allocated() / (1024. ** 3)))
    torch.cuda.reset_peak_memory_stats()

    diff = torch.abs(flow - flow2)
    print('diff max: {}'.format(diff.max()))
    print('diff mean: {}'.format(diff.mean()))
    print('diff median: {}'.format(diff.median()))
