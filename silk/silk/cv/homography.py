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
        rot_matrix = rot_matrix.view(self.batch_size, 3, 3).contiguous()

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

    @staticmethod
    def pixel2cam(depth, intrinsics_inv, points, shape = None):
        # global pixel_coords
        """Transform coordinates in the pixel frame to the camera frame.
        Args:
            depth: depth maps -- [B, H, W]
            intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
        Returns:
            array of (u,v,1) cam coordinates -- [B, 3, H, W]
        """
        
        # points: x, y, z(=1)
        current_pixel_coords = points.clone()  # [B, 3, H*W]
        # print(current_pixel_coords.shape) torch.Size([1, 3, 43780])
        if len(intrinsics_inv.shape)==2:
            intrinsics_inv = intrinsics_inv.unsqueeze(0)
        cam_coords = (intrinsics_inv @ current_pixel_coords)
        # print("in pixel2cam", cam_coords.shape, depth.shape)
        # ([1, 3, 21316]) torch.Size([1, 376, 1241])
        # ([1, 3, 21316]) torch.Size([1, 1200, 1920])

        # print(depth.shape, shape)
        # torch.Size([1, 376, 1241]) (tensor([3]), tensor([376]), tensor([1241]))
        # torch.Size([1, 1200, 1920]) (tensor([3]), tensor([1200]), tensor([1920]))

        
        #########################################
        ### grid sample depth 
        points_ = points[:,:2,:].permute(0,2,1).double()
        # print(points_.shape)
        # torch.Size([1, 43780, 2])
        # print(min(points_[0,:,0]), min(points_[0,:,1]),max(points_[0,:,0]), max(points_[0,:,1]))

        # print(shape)
        # print(depth.shape)
        # torch.Size([1, 3, 370, 1226])
        # torch.Size([1, 370, 1226])

        # normalize bf grid sample
        #shape: c, h, w
        h = shape[2]
        w = shape[3]
        points_[:,:,0] = points_[:,:,0]/(w/2) - 1
        points_[:,:,1] = points_[:,:,1]/(h/2) - 1
        
        # print(min(points_[0,:,0]), min(points_[0,:,1]),max(points_[0,:,0]), max(points_[0,:,1]))
        # tensor(-0.9845, device='cuda:1', dtype=torch.float64) tensor(-0.9486, device='cuda:1', dtype=torch.float64) tensor(0.9845, device='cuda:1', dtype=torch.float64) tensor(0.9054, device='cuda:1', dtype=torch.float64)

        # print(depth.dtype, points_.dtype)
        # torch.float32 torch.float32
        # fuck x,y matters
        if depth.reshape(-1).shape != points.reshape(-1).shape:
            depth = grid_sample(depth.unsqueeze(0), points_.unsqueeze(0), align_corners=False, mode="nearest")        
            
        depth_ = depth.reshape(1, -1).unsqueeze(0)
        # print(depth_.shape)
        # torch.Size([1, 1, 7001])


        # when get_corr
        
        # print(cam_coords.shape)
        # print(depth_.shape, min(depth_[0,0]), max(depth_[0,0]))


        # #########################################
        # ### just getting depth 
        # # points_ = points[:,:2,:].permute(0,2,1).double()
        # # print(points_.shape)
        # # torch.Size([1, 43780, 2])
        # # print(min(points_[0,:,0]), min(points_[0,:,1]),max(points_[0,:,0]), max(points_[0,:,1]))
        # # tensor(9.5000, device='cuda:1', dtype=torch.float64) tensor(9.5000, device='cuda:1', dtype=torch.float64) 
        # # tensor(1216.5000, device='cuda:1', dtype=torch.float64) tensor(360.5000, device='cuda:1', dtype=torch.float64)


        # # print(shape)
        # # print(depth.shape)
        # # torch.Size([1, 3, 370, 1226])
        # # torch.Size([1, 370, 1226])

        # # depth_points_x_start = torch.floor(min(points[0,0,:])).to(torch.int)
        # # depth_points_x_end = torch.floor(max(points[0,0,:])).to(torch.int)
        # # depth_points_y_start = torch.floor(min(points[0,1,:])).to(torch.int)
        # # depth_points_y_end = torch.floor(max(points[0,1,:])).to(torch.int)
        # depth_points_x_start = torch.floor(points[0,0,0]).to(torch.int)
        # depth_points_x_end = torch.floor(points[0,0,-1]).to(torch.int) + 1
        # depth_points_y_start = torch.floor(points[0,1,0]).to(torch.int)
        # depth_points_y_end = torch.floor(points[0,1,-1]).to(torch.int) + 1
        
        # # print(depth_points_x_start, depth_points_x_end, depth_points_y_start, depth_points_y_end)
        # # tensor(9, device='cuda:1', dtype=torch.int32) tensor(1216, device='cuda:1', dtype=torch.int32) tensor(9, device='cuda:1', dtype=torch.int32) tensor(360, device='cuda:1', dtype=torch.int32)
        

        # depth = depth[:, depth_points_y_start:depth_points_y_end, depth_points_x_start:depth_points_x_end]
        # # print(depth.shape)
        # # torch.Size([1, 352, 1208])
        # depth_ = depth.reshape(1, -1).unsqueeze(0)
        # # print(depth_.shape)
        # # torch.Size([1, 1, 425216])



        mask_ = torch.where(depth_>0.003, depth_, 0)
        # print(mask_.shape)
        # # torch.Size([1, 1, 21316])
        mask_tmp = torch.where(depth_>0.003, depth_, 1)
        mask = torch.cat([mask_, mask_, mask_tmp], dim =1)
        # print(mask.shape, cam_coords.shape)
        # torch.Size([1, 3, 425216]) torch.Size([1, 3, 425216])
        
        cam_coords = mask * cam_coords # elementwise mult
        # print(cam_coords.shape) torch.Size([1, 3, 425216])

        return cam_coords #, mask_.reshape(shape)



    @staticmethod #cam_coords, pose_mat, intrinsics, normalize
    def cam2pixel_forward(cam_coords, pose_mat, intrinsics, normalize:bool = False, shape=None):
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
        # print(cam_coords_flat.shape)
        if rot is not None:
            pcoords = rot @ cam_coords_flat
        else:
            pcoords = cam_coords_flat
        # print(pcoords.shape)
        if tr is not None:
            pcoords = pcoords + tr  # [B, 3, H*W]
        # print(pcoords.shape)        
        # go back to pixel coordinate and still homogeneous coordinate
        pcoords = intrinsics @ pcoords
        # print(pcoords.shape)
        X = pcoords[:, 0]
        Y = pcoords[:, 1]
        Z = pcoords[:, 2].clamp(min=1e-3)
        
        # print(max(X))
        # print(max(Y))
        # tensor([ 185.0799,  204.4476,  220.4772,  ..., 6750.4443, 6782.9245,
        # 6627.3625], device='cuda:1', dtype=torch.float64)
        # tensor([  64.3040,   57.3113,   56.2474,  ..., 2028.5731, 2036.7273,
        #         1988.3274], device='cuda:1', dtype=torch.float64)

        # return non-homogeneous coordinate
        if normalize:
            bf_norm = torch.stack([X/Z, Y/Z], dim=2)
            print(shape[-2], shape[-1])
            # tensor([370]) tensor([1226])
            X_norm = 2*(X / Z)/(shape[-1]) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
            Y_norm = 2*(Y / Z)/(shape[-2]) - 1  # Idem [B, H*W]
            # X_norm = 2*(X / Z)/(shape[-1].to(X.device)) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
            # Y_norm = 2*(Y / Z)/(shape[-2].to(X.device)) - 1  # Idem [B, H*W]
        
            pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
            # print(pixel_coords.shape)
            return pixel_coords.float(), bf_norm
        else:
            return torch.stack([X/Z, Y/Z], dim=2)#pixel_coords#.reshape(b, h, w, 2)




    def sj_transform_points(
        self, intrinsics: torch.Tensor, pose_gt: torch.Tensor, depth_map, points, normalize:bool, imshape = None
    ) -> torch.Tensor:

        # input points shape : [1, 3, H*W]
        # output shape: [1, H*W, 2]

        cam_coords = HomographicSampler.pixel2cam(depth_map.unsqueeze(0), intrinsics.inverse(), points, imshape)  # [B,3,H,W] or [B,3n,H,W]
        # print(max(cam_coords[0,0,:]))
        # print(max(cam_coords[0,1,:]))
        # print(max(cam_coords[0,2,:]))
        
        # prev, crop version
        # tensor(-0., device='cuda:1')
        # tensor(0.0094, device='cuda:1')
        # tensor(1., device='cuda:1')

        # now
        # tensor(2851.2961, device='cuda:1', dtype=torch.float64)
        # tensor(1.5257, device='cuda:1', dtype=torch.float64)
        # tensor(10000., device='cuda:1', dtype=torch.float64)

        # Get projection matrix for tgt camera frame to source pixel frame
        # pose_mat = pose_gt[:3,:4].unsqueeze(0).type(torch.cuda.FloatTensor)  # [B,3,4]
        pose_mat = pose_gt[:3,:4].unsqueeze(0).double()
        
        src_pixel_coords = HomographicSampler.cam2pixel_forward(cam_coords.double(), pose_mat, intrinsics.double(), normalize)  # [B,H,W,2]
        # print(max(src_pixel_coords[0,:,0]))
        # print(max(src_pixel_coords[0,:,1]))
        # tensor(1211.9059, device='cuda:1', dtype=torch.float64)
        # tensor(349.9674, device='cuda:1', dtype=torch.float64)
        
        return src_pixel_coords
        


    def transform_points(
        self,
        depth_map,
        pose_gt, 
        intrinsics,
        points: Union[torch.Tensor, List[torch.Tensor]],
        direction: str = "forward",
        ordering: str = "xy",
        imshape = None,
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
        # print(imshape)    
        # torch.Size([1, 21316, 2]) torch.Size([370, 1226])
        # torch.Size([2, 3, 370, 1226])

        # torch.Size([1, 21316, 2]) torch.Size([1200, 1920])
        # (tensor([3]), tensor([1200]), tensor([1920]))
        # pad input if using variable length
        lengths = None
        if not isinstance(points, torch.Tensor):
            lengths = [p.shape[0] for p in points]
            points = pad_sequence(points, batch_first=True)


        # hw, ij order aligns
        # print(points.shape)
        # # torch.Size([1, 21316, 2])
        # print(points[0,:3,0])
        # print(points[0,:3,1])
        # print(crop_point)
        # (tensor([660]), tensor([1582]))
        # definitely row col order
        
        
        # points: x,y order
        # crop point: row,column order
        # points[0,:,0] += crop_point[1]
        # points[0,:,1] += crop_point[0]

        # now points are in hw(row column yet)
        # print(points[0,:3,0])
        # print(points[0,:3,1])


        # nope, already they are xy coord
        # # convert to "xy" ordering
        # if ordering == "yx":
        #     points = points[..., [1, 0]]
            

        # print(points.shape)
        # torch.Size([1, 21316, 2])
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
        # print(sh)
        # torch.Size([1, 21316, 2])

        # points whould be [B, 3, H*W]
        if len(points.shape) ==3:
            points = HomographicSampler._convert_points_to_homogeneous(points)
            # print(points.shape)
            # torch.Size([1, 21316, 3])
            points = points.permute(0,2,1)
            # print(points.shape)
            # torch.Size([1, 3, 21316])
            
        if len(points.shape) ==4:
            # print(points.shape)
            points = points.reshape(self.batch_size, -1, 2)
            points = HomographicSampler._convert_points_to_homogeneous(points)
            # print(points.shape)
            points = points.permute(0,2,1)
            # print(points.shape)

        # print(points.shape)
        # torch.Size([1, 3, 425216])
        # print(max(points[0,0,:]), max(points[0,1,:]), max(points[0,2,:]))
        # tensor(1216.5000, device='cuda:1') tensor(360.5000, device='cuda:1') tensor(1., device='cuda:1')
        
        #input points: xy order
        transformed_points = self.sj_transform_points(intrinsics, pose_gt, depth_map, points, normalize=False, imshape=imshape)
        # print(transformed_points.shape)
        # torch.Size([1, 425216, 2])
       
        if lengths is not None:
            transformed_points = [
                transformed_points[i, :s] for i, s in enumerate(lengths)
            ]

        return transformed_points
