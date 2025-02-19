# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import List, Optional, Tuple, Union
import skimage.io as io
from silk.transforms.tensor import NormalizeRange

import torch
from torch.nn.functional import grid_sample
from torch.nn.utils.rnn import pad_sequence


def resize_homography(
    homography: torch.Tensor,
    original_image_shape: Tuple[int, int],
    new_original_image_shape,
    warped_image_shape=None,
    new_warped_image_shape=None,
) -> torch.Tensor:
    """Change homography matrix when image sizes change.

    Parameters
    ----------
    homography : torch.Tensor
        Homography matrix as a 3x3 Tensor.
    original_image_shape : Tuple[int, int]
        Size of the original image the current homography applies to.
    new_original_image_shape : Tuple[int, int]
        Size of the new original image the resized homography should apply to.
    warped_image_shape : Tuple[int, int], optional
        Size of the warped image the current homography applies to, by default None. Set to `original_image_shape` when None.
    new_warped_image_shape : Tuple[int, int], optional
        Size of the new warped image the resized homography should apply to, by default None. Set to `new_original_image_shape` when None.

    Returns
    -------
    torch.Tensor
        New homography operating on provided image sizes.
    """
    warped_image_shape = (
        original_image_shape if warped_image_shape is None else warped_image_shape
    )
    new_warped_image_shape = (
        new_original_image_shape
        if new_warped_image_shape is None
        else new_warped_image_shape
    )

    # compute resizing factors
    oh_factor = original_image_shape[0] / new_original_image_shape[0]
    ow_factor = original_image_shape[1] / new_original_image_shape[1]

    wh_factor = new_warped_image_shape[0] / warped_image_shape[0]
    ww_factor = new_warped_image_shape[1] / warped_image_shape[1]

    # build resizing diagonal matrices
    up_scale = torch.diag(
        torch.tensor(
            [ow_factor, oh_factor, 1.0],
            dtype=homography.dtype,
            device=homography.device,
        )
    )
    down_scale = torch.diag(
        torch.tensor(
            [ww_factor, wh_factor, 1.0],
            dtype=homography.dtype,
            device=homography.device,
        )
    )

    # apply resizing to homography
    homography = down_scale @ homography @ up_scale

    return homography

# self.forward_matrices is the main
class HomographicSampler:
    """Samples multiple homographic crops from multiples batched images.

    This sampler makes it very easy to sample homographic crops from multiple images by manipulating a virtual crop initially centered on the entire image.
    Applying successive simple transformations (xyz-rotation, shift, scale) will modify the position and shape of that virtual crop.
    Transformations operates on normalized coordinates independent of an image shape.
    The initial virtual crop has its top-left position at (-1, -1), and bottom-right position at (+1, +1).
    Thus the center being at position (0, 0).

    Examples
    --------

    ```python
    hc = HomographicSampler(2, "cpu") # homographic sampler with 2 virtual crops

    hc.scale(0.5) # reduce all virtual crops size by half
    hc.shift(((-0.25, -0.25), (+0.25, +0.25))) # shift first virtual crop to top-left part, second virtual crop to bottom-right part
    hc.rotate(3.14/4., axis="x", clockwise=True, local_center=True) # rotate both virtual crops locally by 45 degrees clockwise (around x-axis)

    crops = hc.extract_crop(image, (100, 100)) # extract two homographic crops defined earlier as (100, 100) images
    ```

    """

    _DEST_COORD = torch.tensor(
        [
            [-1.0, -1.0],  # top-left
            [+1.0, -1.0],  # top-right
            [-1.0, +1.0],  # bottom-left
            [+1.0, +1.0],  # bottom-right
        ],
        dtype=torch.double,
    )

    _VALID_AXIS = {"x", "y", "z"}
    _VALID_DIRECTIONS = {"forward", "backward"}
    _VALID_ORDERING = {"xy", "yx"}

    def __init__(self, batch_size: int, device: str):
        """

        Parameters
        ----------
        batch_size : int
            Number of virtual crops to handle.
        device : str
            Device on which operations will be done.
        """

        self.reset(batch_size, device)

    @staticmethod
    def _convert_points_from_homogeneous(
        points: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """Function that converts points from homogeneous to Euclidean space."""

        # we check for points at max_val
        z_vec: torch.Tensor = points[..., -1:]

        # set the results of division by zeror/near-zero to 1.0
        # follow the convention of opencv:
        # https://github.com/opencv/opencv/pull/14411/files
        mask: torch.Tensor = torch.abs(z_vec) > eps
        scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

        return scale * points[..., :-1]

    @staticmethod
    def _convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
        """Function that converts points from Euclidean to homogeneous space."""

        return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)

    @staticmethod
    def _transform_points(
        trans_01: torch.Tensor, points_1: torch.Tensor
    ) -> torch.Tensor:
        """Function that applies a linear transformations to a set of points."""
        # print("in _transform_points ", trans_01.shape)
        # in _transform_points  torch.Size([1, 1, 1, 3, 3])

        # print("in _transform_points ", trans_01)
        # in _transform_points  tensor([[[[[ 7.2797e-01,  4.0495e-02,  0.0000e+00],
        # [-2.6342e-02,  5.7839e-01,  4.3608e-18],
        # [ 3.2213e-02, -4.4215e-01,  1.0000e+00]]]]], device='cuda:1',
        # dtype=torch.float64)

        points_1 = points_1.to(trans_01.device)
        points_1 = points_1.to(trans_01.dtype)

        # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
        shape_inp = points_1.shape
        points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
        trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1]) # homography? just a matmul?
        # We expand trans_01 to match the dimensions needed for bmm

        # print("in _transform_points ", trans_01)
        # in _transform_points  tensor([[[ 7.2797e-01,  4.0495e-02,  0.0000e+00],
        # [-2.6342e-02,  5.7839e-01,  4.3608e-18],
        # [ 3.2213e-02, -4.4215e-01,  1.0000e+00]]], device='cuda:1',
        # dtype=torch.float64)


        # We expand trans_01 to match the dimensions needed for bmm
        print("in _transform_points ", points_1.shape[0])
        # in _transform_points  164

        trans_01 = torch.repeat_interleave(
            trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0
        )
        print("in _transform_points ", trans_01.shape)
        # in _transform_points  torch.Size([164, 3, 3])

        # to homogeneous
        points_1_h = HomographicSampler._convert_points_to_homogeneous(
            points_1
        )  # BxNxD+1
        # transform coordinates

        print("in _transform_points ", trans_01.permute(0,2,1))
        #in _transform_points  tensor([[[ 7.2797e-01, -2.6342e-02,  3.2213e-02],
        # [ 4.0495e-02,  5.7839e-01, -4.4215e-01],
        # [ 0.0000e+00,  4.3608e-18,  1.0000e+00]],

        points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
        points_0_h = torch.squeeze(points_0_h, dim=-1)
        # to euclidean
        points_0 = HomographicSampler._convert_points_from_homogeneous(
            points_0_h
        )  # BxNxD
        # reshape to the input shape
        points_0 = points_0.reshape(shape_inp)
        return points_0
    




    @staticmethod
    def _create_meshgrid(
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        normalized: bool = True,
    ) -> torch.Tensor:
        """Generate a coordinate grid for an image."""
        if normalized:
            min_x = -1.0
            max_x = +1.0
            min_y = -1.0
            max_y = +1.0
        else:
            min_x = 0.5
            max_x = width - 0.5
            min_y = 0.5
            max_y = height - 0.5

        xs: torch.Tensor = torch.linspace(
            min_x,
            max_x,
            width,
            device=device,
            dtype=dtype,
        )
        ys: torch.Tensor = torch.linspace(
            min_y,
            max_y,
            height,
            device=device,
            dtype=dtype,
        )

        # generate grid by stacking coordinates
        base_grid: torch.Tensor = torch.stack(
            torch.meshgrid([xs, ys], indexing="ij"), dim=-1
        )  # WxHx2
        return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2
    
    @staticmethod
    def sj164_create_meshgrid(
        y: int,
        x: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        normalized: bool = True,
    ) -> torch.Tensor:
        """Generate a coordinate grid for an image."""
        min_x = x + 0.5
        max_x = x + width - 0.5
        min_y = y + 0.5
        max_y = y + height - 0.5

        xs: torch.Tensor = torch.linspace(
            min_x,
            max_x,
            width,
            device=device,
            dtype=dtype,
        )
        ys: torch.Tensor = torch.linspace(
            min_y,
            max_y,
            height,
            device=device,
            dtype=dtype,
        )

        # generate grid by stacking coordinates
        base_grid: torch.Tensor = torch.stack(
            torch.meshgrid([xs, ys], indexing="ij"), dim=-1
        )  # WxHx2
        return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2



    def reset(self, batch_size: Optional[int] = None, device: Optional[str] = None):
        """Resets all the crops to their initial position and sizes.

        Parameters
        ----------
        batch_size : int, optional
            Number of virtual crops to handle, by default None.
        device : str, optional
            Device on which operations will be done, by default None.
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        device = self.device if device is None else device

        self._dest_coords = HomographicSampler._DEST_COORD.to(device)
        self._dest_coords = self._dest_coords.unsqueeze(0)
        self._dest_coords = self._dest_coords.expand(batch_size, -1, -1)

        self._homog_src_coords = HomographicSampler._convert_points_to_homogeneous(
            self._dest_coords
        )

        self._clear_cache()

    def _clear_cache(self):
        """Intermediate data are cached such that the same homographic sampler can efficiently be called several times using the same homographic transforms."""
        self._src_coords = None
        self._forward_matrices = None
        self._backward_matrices = None

    def _to(self, device, name):
        attr = getattr(self, name)
        if attr is not None:
            setattr(self, name, attr.to(device))

    def to(self, device: str):
        """Moves all operations to new device.

        Parameters
        ----------
        device : str
            Pytorch device.
        """
        if device != self.device:
            self._to(device, "_dest_coords")
            self._to(device, "_src_coords")
            self._to(device, "_homog_src_coords")
            self._to(device, "_forward_matrices")
            self._to(device, "_backward_matrices")
        return self

    @property
    def batch_size(self):
        return self._homog_src_coords.shape[0]

    @property
    def device(self):
        return self._homog_src_coords.device

    @property
    def dtype(self):
        return self._homog_src_coords.dtype

    @property
    def src_coords(self) -> torch.Tensor:
        """Coordinates of the homographic crop corners in the virtual image coordinate reference system.
        Those four points are ordered as : (top-left, top-right, bottom-left, bottom-right)

        Returns
        -------
        torch.Tensor
            :math:`(B, 4, 2)` tensor containing the homographic crop foud corners coordinates.
        """
        if self._src_coords is None:
            self._src_coords = HomographicSampler._convert_points_from_homogeneous(
                self._homog_src_coords
            )
        return self._src_coords

    @property
    def dest_coords(self) -> torch.Tensor:
        return self._dest_coords

    def _auto_expand(self, input, outer_dim_size=None, **kwargs):
        """Auto-expand scalar or iterables to be batched."""
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, **kwargs)

        # scalar
        if len(input.shape) == 0:
            input = input.unsqueeze(0)
            if outer_dim_size is None:
                outer_dim_size = 1
            else:
                input = input.expand(outer_dim_size)

        # vector
        if len(input.shape) == 1:
            if outer_dim_size is None:
                outer_dim_size = input.shape[0]
            elif outer_dim_size != input.shape[0]:
                raise RuntimeError(
                    f"provided outer dim size {outer_dim_size} doesn't match input shape {input.shape}"
                )

            input = input.unsqueeze(0)
            input = input.expand(self.batch_size, -1)

        if len(input.shape) != 2:
            raise RuntimeError(f"input should have size BxD (shape is {input.shape}")

        input = input.type(self.dtype)
        input = input.to(self.device)

        return input

    def rotate(
        self,
        angles: Union[float, torch.Tensor],
        clockwise: bool = False,
        axis: str = "z",
        local_center: bool = False,
    ):
        """Rotate virtual crops.

        Parameters
        ----------
        angles : Union[float, torch.Tensor]
            Angles of rotation. If scalar, applied to all crops. If :math:`(B, 1)` tensor, applied to each crop independently.
        clockwise : bool, optional
            Rotational direction, by default False
        axis : str, optional
            Axis of rotation, by default "z". Valid values are "x", "y" and "z". "z" is in-plane rotation. "x" and "y" are out-of-plane rotations.
        local_center : bool, optional
            Rotate on the center of the crop, by default False. If False, use global center of rotation (i.e. initial crop center). This option is only relevant after a shift has been used.

        Raises
        ------
        RuntimeError
            Raised if provided axis is invalid.
        """
        if axis not in HomographicSampler._VALID_AXIS:
            raise RuntimeError(
                f'provided axis "{axis}" isn\'t valid, should be one of {HomographicSampler._VALID_AXIS}'
            )

        angles = self._auto_expand(angles, outer_dim_size=1)

        if clockwise:
            angles = -angles

        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        _1 = torch.ones_like(cos_a)
        _0 = torch.zeros_like(cos_a)

        if axis == "z":
            flatmat = [+cos_a, -sin_a, _0, +sin_a, +cos_a, _0, _0, _0, _1]
        elif axis == "y":
            flatmat = [+cos_a, _0, -sin_a, _0, _1, _0, +sin_a, _0, +cos_a]
        elif axis == "x":
            flatmat = [_1, _0, _0, _0, +cos_a, +sin_a, _0, -sin_a, +cos_a]

        rot_matrix = torch.cat(flatmat, dim=-1)
        rot_matrix = rot_matrix.view(self.batch_size, 3, 3)

        self._clear_cache()

        if local_center:
            center = torch.mean(self._homog_src_coords, dim=1, keepdim=True)

            self._homog_src_coords -= center
            self._homog_src_coords = self._homog_src_coords @ rot_matrix
            self._homog_src_coords += center
        else:
            if axis != "z":
                self._homog_src_coords[..., -1] -= 1.0
            self._homog_src_coords = self._homog_src_coords @ rot_matrix
            if axis != "z":
                self._homog_src_coords[..., -1] += 1.0

    def shift(self, delta: Union[float, Tuple[float, float], torch.Tensor]):
        """Shift virtual crops.

        Parameters
        ----------
        delta : Union[float, Tuple[float, float], torch.Tensor]
            Shift values. Scalar or Tuple will be applied to all crops. :math:`(B, 2)` tensors will be applied to each crop independently.
        """

        delta = self._auto_expand(delta, outer_dim_size=2)
        delta = delta.unsqueeze(1)
        delta = delta * self._homog_src_coords[..., -1].unsqueeze(-1)

        self._clear_cache()
        self._homog_src_coords[..., :2] += delta

    def scale(
        self,
        factors: Union[float, Tuple[float, float], torch.Tensor],
        local_center: bool = False,
    ):
        """Scale the virtual crops.

        Parameters
        ----------
        factors : Union[float, Tuple[float, float], torch.Tensor]
            Scaling factors. Scalar or Tuple will be applied to all crops. :math:`(B, 2)` tensors will be applied to each crop independently.
        local_center : bool, optional
            Scale on the center of the crop, by default False. If False, use global center of rotation (i.e. initial crop center). This option is only relevant after a shift has been used.
        """
        factors = self._auto_expand(factors, outer_dim_size=2)
        factors = factors.unsqueeze(1)

        self._clear_cache()

        if local_center:
            center = torch.mean(self._homog_src_coords, dim=1, keepdim=True)

            self._homog_src_coords -= center
            self._homog_src_coords[..., :2] *= factors
            self._homog_src_coords += center
        else:
            self._homog_src_coords[..., :2] *= factors

    def extract_crop(
        self,
        depth_map,
        intrinsics,
        pose_gt,
        images: torch.Tensor,
        sampling_size: Tuple[int, int],
        direction="forward",
        mode="bilinear",
        padding_mode="zeros",
    ) -> torch.Tensor:
        """Extract all crops from a set of images.

        It can handle one-image-to-many-crops and many-images-to-many-crops.
        If the number of images is smaller than the number of crops, a number of n crops will be asssigned to each image such that :math:`n_crops = n * n_images`.

        Parameters
        ----------
        images : torch.Tensor
            Tensor containing all images (valid shapes are :math:`(B,C,H,W)` and :math:`(C,H,W)`).
        sampling_size : Tuple[int, int]
            Spatial shape of the output crops.
        mode : str, optional
            Sampling mode passed to `grid_sample`, by default "bilinear".
        padding_mode : str, optional
            Padding mode passed to `grid_sample`, by default "zeros".
        direction : str, optional
            Direction of the crop sampling (`src -> dest` or `dest -> src`), by default "forward". Valid are "forward" and "backward".

        Returns
        -------
        torch.Tensor
            Sampled crops using transformed virtual crops.

        Raises
        ------
        RuntimeError
            Raised is `images` shape is invalid.
        RuntimeError
            Raised is `images` batch size isn't a multiple of the number of virtual crops.
        RuntimeError
            Raised is `direction` is invalid.
        """
        print("in homography.py ", images.shape, sampling_size) # [1,3,164,164], [164, 164]
        if images.dim() == 3:
            images = images.unsqueeze(0)
        elif images.dim() != 4:
            raise RuntimeError("provided image(s) should be of shape BxCxHxW or CxHxW")

        if self.batch_size % images.shape[0] != 0:
            raise RuntimeError(
                f"the sampler batch size ({self.batch_size}) should be a multiple of the image batch size (found {images.shape[0]})"
            )

        if direction not in HomographicSampler._VALID_DIRECTIONS:
            raise RuntimeError(
                f'invalid direction "{direction}" found, should be one of {self._VALID_DIRECTIONS}'
            )

        # reshape images to handle multiple crops
        crop_per_image = self.batch_size // images.shape[0]
        images = images.unsqueeze(1)
        images = images.expand(-1, crop_per_image, -1, -1, -1)
        images = images.reshape(self.batch_size, *images.shape[2:])
        
        grid = HomographicSampler.sj164_create_meshgrid(
            # crop_loc[0],
            # crop_loc[1],
            0,0,
            sampling_size[0],
            sampling_size[1],
            device="cuda:1",
            dtype=torch.float64,
            normalized=False,
        )
        grid = grid.expand(self.batch_size, -1, -1, -1)

        # grid = HomographicSampler._transform_points(pose_gt, grid)
        pose_gt = pose_gt.type(torch.float64)
        intrinsics = intrinsics.type(torch.float64)

        # print("warp test ", grid.shape)
        # print(grid)
        # warp test  torch.Size([1, 128, 416, 2])
        # tensor([[[[  0.5000,   0.5000],
        # [  1.5000,   0.5000],

        sh = grid.shape
        # points whould be [B, 3, H*W]
        if len(grid.shape) ==3:
            grid = HomographicSampler._convert_points_to_homogeneous(grid)
        if len(grid.shape) ==4:
            grid = grid.reshape(self.batch_size, -1, 2)
            grid = HomographicSampler._convert_points_to_homogeneous(grid)
            grid = grid.permute(0,2,1)
        # print("warrp test ", grid.shape)

        if direction == "forward":
            grid = self.sj_transform_points(intrinsics, pose_gt, depth_map, grid, normalize=True, direction="forward")
            
        elif direction == "backward":
            grid = self.sj_transform_points(intrinsics, pose_gt, depth_map, grid, normalize=True, direction="backward")
        
        # print("warp test ", grid.shape)
        # print(grid)
        grid = grid.reshape(sh).type_as(images)
        # print("warp test ", grid.shape)
        # print(grid)
        # print("===============================")
        # warrp test  torch.Size([1, 3, 53248])
        # warp test  torch.Size([1, 53248, 2])
        # tensor([[[ 85.2926,  24.0629],
        #         [ 85.8847,  24.0635],
        #         [ 86.4768,  24.0642],
        #         ...,
        #         [330.6962,  99.9657],
        #         [331.2931,  99.9671],
        #         [331.8900,  99.9685]]], device='cuda:1', dtype=torch.float64)

        # warp test  torch.Size([1, 128, 416, 2])
        # [[ 85.2118,  98.2144],
        # [ 85.8037,  98.2157],
        # [ 86.3957,  98.2171],
        # ...,
        # [330.6987,  98.7747],
        # [331.2956,  98.7761],
        # [331.8925,  98.7775]],

        # [[ 85.2112,  98.8074],
        # [ 85.8031,  98.8088],
        # [ 86.3950,  98.8101],
        # ...,
        # [330.6974,  99.3702],
        # [331.2943,  99.3716],
        # [331.8912,  99.3730]],

        # [[ 85.2105,  99.4005],
        # [ 85.8024,  99.4018],
        # [ 86.3944,  99.4032],
        # ...,
        # [330.6962,  99.9657],
        # [331.2931,  99.9671],
        # [331.8900,  99.9685]]]], device='cuda:1')
        # ===============================

        

        # print("warping test", grid.shape) #warping test torch.Size([1, 164, 164, 2])
        # print("warping test ", grid)
        return grid_sample(
            images,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        )

    @staticmethod
    # def set_id_grid(depth):
    #     global pixel_coords
    #     b, h, w = depth.size()
    #     i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    #     j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    #     ones = torch.ones(1, h, w).type_as(depth)

    #     pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
    
    # pixel_coords = None

    @staticmethod
    def pixel2cam(depth, intrinsics_inv, points):
        # global pixel_coords
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps -- [B, H, W]
            intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
        Returns:
            array of (u,v,1) cam coordinates -- [B, 3, H, W]
        """
        b, h, w = depth.size()
        current_pixel_coords = points  # [B, 3, H*W]
        if len(intrinsics_inv.shape)==2:
            intrinsics_inv = intrinsics_inv.unsqueeze(0)
        cam_coords = (intrinsics_inv @ current_pixel_coords)#.reshape(b, -1, h, w)
        # print("in pixel2cam", cam_coords.shape, depth.shape)
        # in pixel2cam torch.Size([1, 3, 53248])
        depth = depth.reshape(1, -1).unsqueeze(0)
        # print(depth.shape) # torch.Size([1, 1, 53248])
        mask = torch.where(depth!=0, depth, 0)
        # print(mask.shape) #torch.Size([1, 1, 53248])
        # print(mask) #tensor([[[1., 1., 1.,  ..., 1., 1., 1.]]], device='cuda:1',
        mask_ = mask.repeat(1, 3, 1)
        # print(mask.shape) # torch.Size([1, 3, 53248])
        cam_coords = mask_ * cam_coords # elementwise mult
        # print(cam_coords.shape) #torch.Size([1, 3, 53248])
        # print(cam_coords[:,:,:3])
        

        return cam_coords, mask.reshape(h, w)

    @staticmethod
    def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, normalize:bool = False):
        """Transform coordinates in the camera frame to the pixel frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
            proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
            proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """
        # b, _, h, w = cam_coords.size()
        cam_coords_flat = cam_coords.reshape(1, 3, -1)  # [B, 3, H*W]
        if proj_c2p_rot is not None:
            pcoords = proj_c2p_rot @ cam_coords_flat
        else:
            pcoords = cam_coords_flat

        if proj_c2p_tr is not None:
            pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2].clamp(min=1e-3)

        if normalize:
            X_norm = 2*(X / Z)/(416-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
            Y_norm = 2*(Y / Z)/(128-1) - 1  # Idem [B, H*W]
        
            pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
            return pixel_coords
        else:
            return torch.stack([X/Z, Y/Z], dim=2)#pixel_coords#.reshape(b, h, w, 2)


    @staticmethod #cam_coords, pose_mat, intrinsics, normalize
    def cam2pixel_forward(cam_coords, pose_mat, intrinsics, normalize:bool = False):
        """Transform coordinates in the camera frame to the pixel frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
            proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
            proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """

        rot, tr = pose_mat[..., :3], pose_mat[..., -1:] #[B, 3, 3], [B, 3, 1]

        # b, _, h, w = cam_coords.size()
        cam_coords_flat = cam_coords.reshape(1, 3, -1)  # [B, 3, H*W]
        if rot is not None:
            pcoords = rot @ cam_coords_flat
        else:
            pcoords = cam_coords_flat
        if tr is not None:
            pcoords = pcoords + tr  # [B, 3, H*W]
        
        # go back to pixel coordinate and still homogeneous coordinate
        pcoords = intrinsics @ pcoords
        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2].clamp(min=1e-3)
        
        # return non-homogeneous coordinate
        if normalize:
            X_norm = 2*(X / Z)/(416) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
            Y_norm = 2*(Y / Z)/(128) - 1  # Idem [B, H*W]
        
            pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
            return pixel_coords
        else:
            return torch.stack([X/Z, Y/Z], dim=2)#pixel_coords#.reshape(b, h, w, 2)



    @staticmethod #cam_coords, pose_mat, intrinsics, normalize
    def cam2pixel_backward(cam_coords, pose_mat, intrinsics, normalize:bool = False):
        """Transform coordinates in the camera frame to the pixel frame.
        Args:
            cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
            proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
            proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
        Returns:
            array of [-1,1] coordinates -- [B, 2, H, W]
        """

        rot, tr = pose_mat[..., :3], pose_mat[..., -1:] #[B, 3, 3], [B, 3, 1]

        # b, _, h, w = cam_coords.size()
        cam_coords_flat = cam_coords.reshape(1, 3, -1)  # [B, 3, H*W]
        
        if tr is not None:
            pcoords = cam_coords_flat - tr  # [B, 3, H*W]
        
        if rot is not None:
            pcoords = rot.inverse() @ pcoords
        else:
            pcoords = pcoords
        
        # go back to pixel coordinate and still homogeneous coordinate
        pcoords = intrinsics @ pcoords
        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2].clamp(min=1e-3)
        
        # return non-homogeneous coordinate
        if normalize:
            X_norm = 2*(X / Z)/(416-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
            Y_norm = 2*(Y / Z)/(128-1) - 1  # Idem [B, H*W]
        
            pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
            return pixel_coords
        else:
            return torch.stack([X/Z, Y/Z], dim=2)#pixel_coords#.reshape(b, h, w, 2)



    # def sj_transform_points_original(
    #     self, intrinsics: torch.Tensor, pose_gt: torch.Tensor, depth_map, points, normalize:bool, direction: str
    # ) -> torch.Tensor:
    #     # batch_size, _, img_height, img_width = depth_map.shape
    #     # points: [1, 3, H, W]
    #     print("in sj_transform_points",intrinsics.shape, depth_map.shape, points.shape)
    #     cam_coords, mask = HomographicSampler.pixel2cam(depth_map.unsqueeze(0), intrinsics.inverse(), points)  # [B,3,H,W] or [B,3n,H,W]
    #     print("cam coord", cam_coords)
    #     # in sj_transform_points torch.Size([3, 3]) torch.Size([128, 416]) torch.Size([1, 3, 53248])
    #     # cam coord tensor([[[-0.8427, -0.8386, -0.8345,  ...,  0.8662,  0.8703,  0.8744],
    #     #         [-0.2375, -0.2375, -0.2375,  ...,  0.2781,  0.2781,  0.2781],
    #     #         [ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000]]],
    #     #     device='cuda:1', dtype=torch.float64)

    #     pose_mat = pose_gt[:3,:4].unsqueeze(0)  # [B,3,4]
    #     # if direction == "backward":
    #     #     pose_mat[:3, :3] = 
    #     # Get projection matrix for tgt camera frame to source pixel frame
    #     proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]
    #     rot, tr = proj_cam_to_src_pixel[..., :3], proj_cam_to_src_pixel[..., -1:]
    #     # rot, tr = proj_cam_to_src_pixel[:,:3, :3], proj_cam_to_src_pixel[:,:3, 3]
    #     src_pixel_coords = HomographicSampler.cam2pixel(cam_coords, rot, tr, normalize)  # [B,H,W,2]
    #     # projected_img = grid_sample(img, src_pixel_coords, align_corners=True)
    #     print(src_pixel_coords.shape)
    #     # torch.Size([1, 53248, 2]) when extract crop 

    #     # valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
    #     print("out sj_transform")
    #     return src_pixel_coords

    def sj_transform_points(
        self, intrinsics: torch.Tensor, pose_gt: torch.Tensor, depth_map, points, normalize:bool, direction: str
    ) -> torch.Tensor:
        # batch_size, _, img_height, img_width = depth_map.shape
        # points: [1, 3, H, W]
        # print("in sj_transform_points",intrinsics.shape, depth_map.shape, points.shape)
                
        cam_coords, mask = HomographicSampler.pixel2cam(depth_map.unsqueeze(0), intrinsics.inverse(), points)  # [B,3,H,W] or [B,3n,H,W]
        # print("cam coord", cam_coords)
        # in sj_transform_points torch.Size([3, 3]) torch.Size([128, 416]) torch.Size([1, 3, 53248])
        # cam coord tensor([[[-0.8427, -0.8386, -0.8345,  ...,  0.8662,  0.8703,  0.8744],
        #         [-0.2375, -0.2375, -0.2375,  ...,  0.2781,  0.2781,  0.2781],
        #         [ 1.0000,  1.0000,  1.0000,  ...,  1.0000,  1.0000,  1.0000]]],
        #     device='cuda:1', dtype=torch.float64)

        # Get projection matrix for tgt camera frame to source pixel frame
        pose_mat = pose_gt[:3,:4].unsqueeze(0)  # [B,3,4]
        if direction == "backward":
            src_pixel_coords = HomographicSampler.cam2pixel_backward(cam_coords, pose_mat, intrinsics, normalize)  # [B,H,W,2]
        elif direction == "forward":
            src_pixel_coords = HomographicSampler.cam2pixel_forward(cam_coords, pose_mat, intrinsics, normalize)  # [B,H,W,2]
        
        # print(src_pixel_coords.shape)
        # torch.Size([1, 53248, 2]) when extract crop 

        # valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1
        # print("out sj_transform")
        if normalize: # for warp
            return src_pixel_coords
        else: # for get_corr
            return src_pixel_coords, mask


    def transform_points(
        self,
        depth_map,
        pose_gt, 
        intrinsics,
        points: Union[torch.Tensor, List[torch.Tensor]],
        image_shape: Optional[Tuple[int, int]] = None,
        direction: str = "forward",
        ordering: str = "xy",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Apply homography to a set of points.

        Parameters
        ----------
        points : Union[torch.Tensor, List[torch.Tensor]]
            BxNx2 tensor or list of Nx2 tensors containing the coordinates to transform.
        image_shape : Optional[Tuple[int, int]], optional
            Shape of the tensor the coordinates references, as in (height, width), by default None.
            If not provided, the coordinates are assumed to be already normalized between [-1, +1].
        direction : str, optional
            Direction of the homography, by default "forward".
        ordering : str, optional
            Specify the order in which the x,y coordinates are stored in "points", by default "xy".

        Returns
        -------
        Union[torch.Tensor, List[torch.Tensor]]
            Transformed coordinates.

        Raises
        ------
        RuntimeError
            If the provided direction is invalid.
        RuntimeError
            If the provided ordering is invalid.
        """
        # check arguments
        if direction not in HomographicSampler._VALID_DIRECTIONS:
            raise RuntimeError(
                f'invalid direction "{direction}" found, should be one of {self._VALID_DIRECTIONS}'
            )
        if ordering not in HomographicSampler._VALID_ORDERING:
            raise RuntimeError(
                f'invalid ordering "{ordering}" found, should be one of {self._VALID_ORDERING}'
            )


        # print("in transform points")
        # print(points.shape, depth_map.shape)

        # pad input if using variable length
        lengths = None
        if not isinstance(points, torch.Tensor):
            lengths = [p.shape[0] for p in points]
            points = pad_sequence(points, batch_first=True)

        # convert to "xy" ordering
        if ordering == "yx":
            points = points[..., [1, 0]]

        # print("in transform points")
        # print(points.shape, depth_map.shape)

        transform_per_points = self.batch_size // points.shape[0]
        points = points.unsqueeze(1)
        points = points.expand(-1, transform_per_points, -1, -1)
        points = points.reshape(self.batch_size, *points.shape[2:])

        # change lengths size accordingly
        if transform_per_points != 1:
            lengths = list(
                itertools.chain.from_iterable(
                    itertools.repeat(s, transform_per_points) for s in lengths
                )
            )
         
        sh = points.shape 
        # points whould be [B, 3, H*W]
        if len(points.shape) ==3:
            points = HomographicSampler._convert_points_to_homogeneous(points)
            points = points.permute(0,2,1)

        if len(points.shape) ==4:
            # print(points.shape)
            points = points.reshape(self.batch_size, -1, 2)
            points = HomographicSampler._convert_points_to_homogeneous(points)
            # print(points.shape)
            points = points.permute(0,2,1)
            # print(points.shape)

        if direction == "forward":
            transformed_points, mask = self.sj_transform_points(intrinsics, pose_gt, depth_map, points, normalize=False, direction="forward")
            
        elif direction == "backward":
            transformed_points, mask = self.sj_transform_points(intrinsics, pose_gt, depth_map, points, normalize=False, direction="backward")


        transformed_points = transformed_points.reshape(sh).to("cuda:1")
        
        # print("in transform_points ", transformed_points.shape)
        # in transform_points  torch.Size([1, 21316, 2])

        # convert back to initial ordering
        if ordering == "yx":
            transformed_points = transformed_points[..., [1, 0]]

        # remove padded results if input was variable length
        if lengths is not None:
            transformed_points = [
                transformed_points[i, :s] for i, s in enumerate(lengths)
            ]

        return transformed_points, mask
