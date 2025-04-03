import math
import numbers
import random
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import _functional_pil as F_pil
from torchvision.transforms import _functional_tensor as F_t
from torchvision.transforms._functional_tensor import _gen_affine_grid, _cast_squeeze_in, _cast_squeeze_out
from torchvision.transforms.functional import (_get_inverse_affine_matrix,
                                               _interpolation_modes_from_int,
                                               _log_api_usage_once,
                                               get_dimensions,
                                               pil_modes_mapping)
from torchvision.transforms.transforms import (_check_sequence_input,
                                               _interpolation_modes_from_int,
                                               _log_api_usage_once,
                                               _setup_angle)


def _get_inverse_affine_matrix(
    center: List[float], angle: float, translate: List[float], scale: float, shear: List[float], inverted: bool = True
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a - sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        if isinstance(scale, Iterable):
            matrix = [x / scale[0] for x in matrix[:3]] + [x / scale[1] for x in matrix[3:]]
        else:
            matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        if isinstance(scale, Iterable):
            matrix = [x * scale[0] for x in matrix[:3]] + [x / scale[1] for x in matrix[3:]]
        else:
            matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix

def affine(
    img: Tensor,
    angle: float,
    translate: List[int],
    scale: Union[float, List[float]],
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
    center: Optional[List[int]] = None,
    return_matrix: bool = False,
) -> Tensor:
    """Apply affine transformation on the image keeping image center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        img (PIL Image or Tensor): image to transform.
        angle (number): rotation angle in degrees between -180 and 180, clockwise direction.
        translate (sequence of integers): horizontal and vertical translations (post-rotation translation)
        scale (float): overall scale
        shear (float or sequence): shear angle value in degrees between -180 to 180, clockwise direction.
            If a sequence is specified, the first value corresponds to a shear parallel to the x-axis, while
            the second value corresponds to a shear parallel to the y-axis.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            The corresponding Pillow integer constants, e.g. ``PIL.Image.BILINEAR`` are accepted as well.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.

            .. note::
                In torchscript mode single int/float value is not supported, please use a sequence
                of length 1: ``[value, ]``.
        center (sequence, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.

    Returns:
        PIL Image or Tensor: Transformed image.
    """
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(affine)

    if isinstance(interpolation, int):
        interpolation = _interpolation_modes_from_int(interpolation)
    elif not isinstance(interpolation, InterpolationMode):
        raise TypeError(
            "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
        )

    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if not isinstance(translate, (list, tuple)):
        raise TypeError("Argument translate should be a sequence")

    if len(translate) != 2:
        raise ValueError("Argument translate should be a sequence of length 2")

    if isinstance(scale, Iterable):
        if any([x <= 0.0 for x in scale]):
            raise ValueError("Argument scale should be positive")
    else:
        if scale <= 0.0:
            raise ValueError("Argument scale should be positive")

    if not isinstance(shear, (numbers.Number, (list, tuple))):
        raise TypeError("Shear should be either a single value or a sequence of two values")

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError(f"Shear should be a sequence containing two values. Got {shear}")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    _, height, width = get_dimensions(img)
    if not isinstance(img, torch.Tensor):
        # center = (width * 0.5 + 0.5, height * 0.5 + 0.5)
        # it is visually better to estimate the center without 0.5 offset
        # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
        if center is None:
            center = [width * 0.5, height * 0.5]
        matrix = _get_inverse_affine_matrix(center, angle, translate, scale, shear)
        pil_interpolation = pil_modes_mapping[interpolation]
        if return_matrix:
            return F_pil.affine(img, matrix=matrix, interpolation=pil_interpolation, fill=fill), matrix
        return F_pil.affine(img, matrix=matrix, interpolation=pil_interpolation, fill=fill)

    center_f = [0.0, 0.0]
    if center is not None:
        _, height, width = get_dimensions(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    translate_f = [1.0 * t for t in translate]
    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)
    if return_matrix:
        return F_t.affine(img, matrix=matrix, interpolation=interpolation.value, fill=fill), matrix
    return F_t.affine(img, matrix=matrix, interpolation=interpolation.value, fill=fill)

class AffineWithGrid(torch.nn.Module):
    def __init__(self, degrees=None, translate=None, scale=None, shear=None, interpolation=InterpolationMode.NEAREST, fill=None, center=None,
                 bound_translate: bool=False):
        super().__init__()
        _log_api_usage_once(self)

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError(f"translation values should be between 0 and 1. Got {t}")
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2, 4))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center
        
        if translate is not None and bound_translate:
            assert (np.array(scale) >= 1.).all()
            assert degrees is None or degrees == [0., 0.]
        self.bound_translate = bound_translate
    
    @staticmethod
    def get_params(
        degrees: List[float],
        translate: Optional[List[float]],
        scale_ranges: Optional[List[float]],
        shears: Optional[List[float]],
        img_size: List[int],
        bound_translate: bool=False
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        if scale_ranges is not None:
            if len(scale_ranges) == 2:
                scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
                w_scale, h_scale = img_size[0] * scale, img_size[1] * scale
            elif len(scale_ranges) == 4:
                scale = (float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item()),
                        float(torch.empty(1).uniform_(scale_ranges[2], scale_ranges[3]).item()))
                w_scale, h_scale = img_size[0] * scale[0], img_size[1] * scale[1]
            else:
                raise ValueError("scale_ranges should be a list of two or four values")
        else:
            scale = 1.0
        
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
            
        if translate is not None:
            for t in translate:
                assert isinstance(t, float) and (0. <= np.abs(t) <= 1.), f"translation values should be between 0 and 1. Got {t}"
        
            if len(translate) == 2:
                max_dx = float(translate[0] * img_size[0])
                max_dy = float(translate[1] * img_size[1])
                if bound_translate:
                    assert scale >= 1. if isinstance(scale, float) else (scale[0] >= 1. and scale[1] >= 1.)
                    max_dx = min(max_dx, (img_size[0] * (scale if isinstance(scale, float) else scale[0]) - img_size[0])//2)
                    max_dy = min(max_dy, (img_size[1] * (scale if isinstance(scale, float) else scale[1]) - img_size[1])//2)
                tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
                ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
                translations = (tx, ty)
            elif len(translate) == 4:
                min_dx, max_dx = float(translate[0] * img_size[0]), float(translate[1] * img_size[0])
                min_dy, max_dy = float(translate[2] * img_size[1]), float(translate[3] * img_size[1])
                if bound_translate:
                    assert scale >= 1. if isinstance(scale, float) else (scale[0] >= 1. and scale[1] >= 1.)
                    bound_x, bound_y = ((w_scale - img_size[0])//2, (h_scale - img_size[1])//2)
                    min_dx, max_dx = np.sign(min_dx) * min(np.abs(min_dx), bound_x), np.sign(max_dx) * min(np.abs(max_dx), bound_x)
                    min_dy, max_dy = np.sign(min_dy) * min(np.abs(min_dy), bound_y), np.sign(max_dy) * min(np.abs(max_dy), bound_y)
                tx = int(round(torch.empty(1).uniform_(min_dx, max_dx).item()))
                ty = int(round(torch.empty(1).uniform_(min_dy, max_dy).item()))
                translations = (tx, ty)
            else:
                raise ValueError("translate should be a list of two or four values")
        else:
            translations = (0, 0)

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear
    
    def forward(self, img, params=None, return_params=False):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        assert len(img.shape) == 3
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        img_size = [width, height]  # flip for keeping BC on get_params call
        if params:
            ret = params
        else:
            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size, self.bound_translate)

        img, matrix = affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center, return_matrix=True)
        
        dtype = img.dtype if torch.is_floating_point(img) else torch.float32
        inv_matrix = torch.Tensor(cv2.invertAffineTransform(np.array(matrix).reshape(2, 3))).reshape(1, 2, 3)
        matrix = torch.tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
        shape = img.shape

        grid = _gen_affine_grid(matrix, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2]).reshape(*shape[1:], 2)
        inv_grid = _gen_affine_grid(inv_matrix, w=shape[-1], h=shape[-2], ow=shape[-1], oh=shape[-2]).reshape(*shape[1:], 2)
        
        if return_params: 
            return img, grid, inv_grid, ret
        return img, grid, inv_grid
    
def _apply_grid_transform(
    img: Tensor, grid: Tensor, mode: str, fill: Optional[Union[int, float, List[float]]]
) -> Tensor:
    """Copy of the function from _functional_tensor, except it acceps batched img and grids.

    Args:
        img (Tensor): _description_
        grid (Tensor): _description_
        mode (str): _description_
        fill (Optional[Union[int, float, List[float]]]): _description_

    Returns:
        Tensor: _description_
    """
    img, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [grid.dtype])

    if img.shape[0] > 1:
        if len(grid.shape) == 3:
            grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])
        else:
            assert img.shape[0] == grid.shape[0]

    # Append a dummy mask for customized fill colors, should be faster than grid_sample() twice
    if fill is not None:
        mask = torch.ones((img.shape[0], 1, img.shape[2], img.shape[3]), dtype=img.dtype, device=img.device)
        img = torch.cat((img, mask), dim=1)
    
    if grid.shape[-3] == 2:
        grid = grid.transpose(-3, -1).transpose(-2, -3) # N,C,H,W -> N,W,H,C -> N,H,W,C

    img = torch.nn.functional.grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    # Fill with required color
    if fill is not None:
        mask = img[:, -1:, :, :]  # N * 1 * H * W
        img = img[:, :-1, :, :]  # N * C * H * W
        mask = mask.expand_as(img)
        fill_list, len_fill = (fill, len(fill)) if isinstance(fill, (tuple, list)) else ([float(fill)], 1)
        fill_img = torch.tensor(fill_list, dtype=img.dtype, device=img.device).view(1, len_fill, 1, 1).expand_as(img)
        if mode == "nearest":
            mask = mask < 0.5
            img[mask] = fill_img[mask]
        else:  # 'bilinear'
            img = img * mask + (1.0 - mask) * fill_img

    img = _cast_squeeze_out(img, need_cast, need_squeeze, out_dtype)
    return img
