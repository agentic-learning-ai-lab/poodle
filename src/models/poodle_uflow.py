import torch 

from src.models.registry import register_model
from src.models.poodle import Poodle
from src.models.uflow import UFlow

MAX_TRIES_PER_CROP = 5

class PoodleUFlow(Poodle):
    def __init__(self, flow_model_checkpoint_uflow, *args, **kwargs):
    
        super(PoodleUFlow, self).__init__(*args, **kwargs)
        del self.flow_model
        # flow model
        self.flow_module = UFlow(
            backbone='pwc',
            inference_res=[256, 512],
            estimator_modules='1_none-4_flow',
            downproj_feat=None,
            pre_corr_norm='norm',
            inter_conv_norm=False,
            refinement_module=True,
            refinement_module_dims='base',
        )
        for param in self.flow_module.parameters():
            param.requires_grad = False
        self.flow_module.eval()
        loaded_flow_model_state_dict = torch.load(flow_model_checkpoint_uflow)
        self.flow_module.load_state_dict(loaded_flow_model_state_dict['model'])
        
        def flow_inf_fn(x1, x2, iters, test_mode):
            return None, self.flow_module(x1, x2)
        self.flow_model = flow_inf_fn

@register_model
def poodle_uflow(**kwargs):
    return poodle_uflow(**kwargs)
