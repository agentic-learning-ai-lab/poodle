from typing import List

import torch
from torchvision.models.resnet import Bottleneck, ResNet, ResNet50_Weights

class ResNetBackbone(ResNet):
    def forward_features(self, x, pool=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if pool:
            x = self.avgpool(x)
            # x = torch.flatten(x, 1)
            # global average pooling, (N, C, H, W) -> (N, C, 1, 1)
        return x
    
    def forward_feature_pyramid(self, x, stages=[3]) -> List[torch.Tensor]:
        feature_pyramid = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            if i in stages:
                feature_pyramid.append(x)
            else:
                feature_pyramid.append(None)

        return feature_pyramid

def resnet50(pretrained=False, **kwargs):
    # resnet50, where we remove downsampling in last two stages and use dilated convolutions instead
    model = ResNetBackbone(block=Bottleneck, layers=[3, 4, 6, 3], replace_stride_with_dilation=[False, False, False], **kwargs)
    if pretrained:
        model.load_state_dict(ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True))
    del model.fc
    return model

def resnet50_stride8(pretrained=False, **kwargs):
    # resnet50, where we remove downsampling in last two stages and use dilated convolutions instead
    model = ResNetBackbone(block=Bottleneck, layers=[3, 4, 6, 3], replace_stride_with_dilation=[False, True, True], **kwargs)
    if pretrained:
        model.load_state_dict(ResNet50_Weights.IMAGENET1K_V2.get_state_dict(progress=True, check_hash=True))
    del model.fc
    return model
