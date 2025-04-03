import torch
from spatial_correlation_sampler import SpatialCorrelationSampler
from torch import nn
from torch.nn import functional as F
from functools import partial

from src.utils.flow import normalize_features, upsample_flow, warp

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

class PWCContext(nn.Module):
    def __init__(self, in_chans, dims=[128, 128, 128, 96, 64, 32], dilation=[1, 2, 4, 8, 16, 1],
                 norm_layer=partial(LayerNorm, eps=1e-6, data_format="channels_first")):
        super().__init__()
        layers = []
        dims = [in_chans] + dims
        for i in range(len(dims) - 1):
            layers.append(nn.Conv2d(in_channels=dims[i], out_channels=dims[i+1], kernel_size=3, stride=1, padding=dilation[i], dilation=dilation[i]))
            layers.append(norm_layer(dims[i+1]))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
        layers.append(nn.Conv2d(in_channels=dims[-1], out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1))
        self.context_module = nn.Sequential(*layers)
    
    def forward(self, features):
        return self.context_module(features)

class PWCFlowPredictor(nn.Module):
    def __init__(self, 
                 encoder_channels=[32, 32, 32, 32, 32], # from default backbone
                 num_levels=5,
                 leaky_relu_alpha=0.1,
                 dropout_p=0.1,
                 flow_module_channels=(128, 128, 96, 64, 32), corr_patch_size=9, context_dim=32,
                 refinement_module=True, refinement_module_dims=[(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]):
        super(PWCFlowPredictor, self).__init__()
        
        self.corr_patch_size = corr_patch_size
        self.leaky_relu_alpha = leaky_relu_alpha
        
        # build flow modules
        flow_modules = [nn.Identity()]
        for i in range(1, num_levels):
            layers = []
            curr_dim = corr_patch_size ** 2 + encoder_channels[i] + (0 if i == 4 else context_dim) + (0 if i == 4 else 2)
            for c in flow_module_channels:
                layers.append(nn.Sequential(
                    nn.Conv2d(curr_dim, c, 3, padding=1, stride=1),
                    nn.LeakyReLU(leaky_relu_alpha, inplace=True,)
                ))
                curr_dim += c
            layers.append(nn.Conv2d(flow_module_channels[-1], 2, 3, padding=1, stride=1))
            flow_modules.append(nn.Sequential(*layers))
        self.flow_modules = nn.ModuleList(flow_modules)
        
        self.corr = SpatialCorrelationSampler(kernel_size=1, patch_size=corr_patch_size, stride=1, dilation_patch=1, padding=0, dilation=1)
        self.corr_relu = nn.LeakyReLU(leaky_relu_alpha, inplace=True)
        
        if dropout_p > 0.:
            self.dropout = nn.Dropout2d(p=dropout_p)
        else:
            self.dropout = None
        
        # build upsample layers
        upsample_modules = [nn.Identity(), nn.Identity()]
        for _ in range(2, num_levels):
            upsample_modules.append(nn.ConvTranspose2d(flow_module_channels[-1], context_dim, 4, padding=1, stride=2))
        self.upsample_modules = nn.ModuleList(upsample_modules)
        
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
            list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[1:]):
            if flow_up is None:
                warped2 = feature2
            else:
                warped2 = warp(feature2, flow_up)
            
            # compute cost volume between feature1 and warped2
            feature1_norm, warped2_norm = normalize_features(
                [feature1, warped2],
                normalize=True, center=True,
                across_channels=True, across_features=True
            )
            
            cost_volume = self.corr(feature1_norm.contiguous(), warped2_norm.contiguous()).flatten(1, 2)/feature1_norm.shape[1]
            cost_volume = self.corr_relu(cost_volume)

            # Compute context and flow from previous flow, cost volume, and features1.
            if flow_up is None:
                x_in = torch.cat([cost_volume, feature1], dim=1)
            else:
                if context_up is None:
                    x_in = torch.cat([flow_up, cost_volume, feature1], dim=1)
                else:
                    x_in = torch.cat([context_up, flow_up, cost_volume, feature1], dim=1)
            x_out = None
            flow_layers = self.flow_modules[level]

            for layer in flow_layers[:-1]:
                x_out = layer(x_in)
                x_in = torch.cat([x_in, x_out], dim=1)
            context = x_out
            flow = flow_layers[-1](context)
            
            # dropout
            if self.dropout:
                flow = self.dropout(flow)
                context = self.dropout(context)
        
            # residual connection
            if flow_up is not None:
                flow += flow_up
            
            # upsample flow and context
            flow_up = upsample_flow(flow, 2.)
            context_up = self.upsample_modules[level](context)
            
            # append
            flows.insert(0, flow)
        
        if self.refinement_module is not None:
            refinement = self.refinement_module(torch.cat([context, flow], dim=1))
            if self.dropout:
                refinement = self.dropout(refinement)
            refined_flow = flow + refinement
            flows[0] = refined_flow
        return flows

class PWCBackbone(nn.Module):
    def __init__(self, input_channels=3, filters=[(3, 32)]*5, leaky_relu_alpha=0.1):
        super(PWCBackbone, self).__init__()
        
        self.blocks = []
        
        # create encoder and modify
        for level, (n_layers, n_filters) in enumerate(filters):
            group = []
            for i in range(n_layers):
                stride = 2 if i == 0 else 1
                conv = nn.Conv2d(input_channels, n_filters, 3, stride=stride, padding=1)
                input_channels = n_filters
                group.append(conv)
                group.append(nn.LeakyReLU(leaky_relu_alpha, inplace=True))
            self.blocks.append(nn.Sequential(*group))
        self.blocks = nn.ModuleList(self.blocks)
        
        self.encoder_channels = [f for _, f in filters]
        self.num_levels = len(filters)
    
    def forward_features(self, x, pool=True):
        return self.forward(x)
    
    def forward_feature_pyramid(self, x):
        feature_pyramid = []
        for level, block in enumerate(self.blocks):
            x = block(x)
            feature_pyramid.append(x)
        return feature_pyramid
    
    def forward(self, x):
        for level, block in enumerate(self.blocks):
            x = block(x)
        return x

if __name__ == "__main__":
    import torch
    model = PWCBackbone()
    print(model)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
    y = model.forward_feature_pyramid(x)
    print([x.shape for x in y])
    # y = model.forward_features(x)
    # print([x.shape for x in y])
                
    flow_model = PWCFlowPredictor()
    print(flow_model)
    flow = flow_model(y, y)
    print([x.shape for x in y])
    print([x.shape for x in flow])