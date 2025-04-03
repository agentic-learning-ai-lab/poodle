import torch
import torch.nn.functional as F

def weighted_ssim(x, y, weight, c1=float('inf'), c2=9e-6, weight_epsilon=0.01):
    """Computes a weighted structured image similarity measure.
    Args:
        x: a batch of images, of shape [B, C, H, W].
        y:  a batch of images, of shape [B, C, H, W].
        weight: shape [B, 1, H, W], representing the weight of each
        pixel in both images when we come to calculate moments (means and
        correlations). values are in [0,1]
        c1: A floating point number, regularizes division by zero of the means.
        c2: A floating point number, regularizes division by zero of the second
        moments.
        weight_epsilon: A floating point number, used to regularize division by the
        weight.

    Returns:
        A tuple of two pytorch Tensors. First, of shape [B, C, H-2, W-2], is scalar
        similarity loss per pixel per channel, and the second, of shape
        [B, 1, H-2. W-2], is the average pooled `weight`. It is needed so that we
        know how much to weigh each pixel in the first tensor. For example, if
        `'weight` was very small in some area of the images, the first tensor will
        still assign a loss to these pixels, but we shouldn't take the result too
        seriously.
    """
    _avg_pool3x3 = torch.nn.AvgPool2d(kernel_size=(3,3), stride=(1,1), padding=0)

    if c1 == float('inf') and c2 == float('inf'):
        raise ValueError('Both c1 and c2 are infinite, SSIM loss is zero. This is '
                            'likely unintended.')
    average_pooled_weight = _avg_pool3x3(weight)
    weight_plus_epsilon = weight + weight_epsilon
    inverse_average_pooled_weight = 1.0 / (average_pooled_weight + weight_epsilon)

    def weighted_avg_pool3x3(z):
        wighted_avg = _avg_pool3x3(z * weight_plus_epsilon)
        return wighted_avg * inverse_average_pooled_weight

    mu_x = weighted_avg_pool3x3(x)
    mu_y = weighted_avg_pool3x3(y)
    sigma_x = weighted_avg_pool3x3(x**2) - mu_x**2
    sigma_y = weighted_avg_pool3x3(y**2) - mu_y**2
    sigma_xy = weighted_avg_pool3x3(x * y) - mu_x * mu_y
    if c1 == float('inf'):
        ssim_n = (2 * sigma_xy + c2)
        ssim_d = (sigma_x + sigma_y + c2)
    elif c2 == float('inf'):
        ssim_n = 2 * mu_x * mu_y + c1
        ssim_d = mu_x**2 + mu_y**2 + c1
    else:
        ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    result = ssim_n / ssim_d
    return torch.clip((1 - result) / 2, 0, 1), average_pooled_weight

if __name__ == "__main__":
    # x = torch.rand(1, 3, 224, 224)
    # y = torch.rand(1, 3, 224, 224)
    # w = torch.rand(1, 1, 224, 224)
    
    # score, weights = weighted_ssim(x, y, w)
    # ssim_ = (score * weights)
    # ssim_ = ssim_.sum(dim=(1, 2, 3)) / (weights.sum(dim=(1, 2, 3)) + 1e-6)
    # print(ssim_)
    # score, weights = weighted_ssim_old(x, y, w, float('inf'), 9e-6, 0.01)
    # ssim_ = (score * weights)
    # ssim_ = ssim_.sum(dim=(1, 2, 3)) / (weights.sum(dim=(1, 2, 3)) + 1e-6)
    # print(ssim_)
    import pdb; pdb.set_trace()
    x = torch.cat((torch.ones(4, 3, 500, 640), torch.zeros(4, 3, 140, 640)), dim=-2)
    y = torch.ones(4, 3, 640, 640) * 0.8
    mask = torch.cat([torch.ones(4, 1, 640, 320), torch.zeros(4, 1, 640, 320)], dim=-1)
    ssim, weights = weighted_ssim(x, y, mask, c1=float('inf'), c2=9e-6, weight_epsilon=0.01)
    
    5/0.
    