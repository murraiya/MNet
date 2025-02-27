import numpy as np
import torch
import skimage.io as io
import matplotlib.pyplot as plt
import torch.nn.functional as F
from silk.matching.mnn import mutual_nearest_neighbor
from silk.cv.homography import HomographicSampler
from silk.losses.info_nce.loss import positions_to_unidirectional_correspondence


def epiploar_loss(intrinsics, pose_mat, kpts_1_, kpts_2_, descriptors):

    print("what the matches do")
    
    matches = mutual_nearest_neighbor(
        descriptors[0].clone(),
        descriptors[1].clone()
    )
   
    kpts_1 = kpts_1_.clone()
    kpts_2 = kpts_2_.clone()

    kpts_2 = (kpts_2[matches[:, 1]])[:, [1,0]]
    kpts_1 = (kpts_1[matches[:, 0]])[:, [1,0]]
   

        
    def veccrossproduct(vector3d: torch.Tensor)->torch.Tensor:
        # sj
        # vec cross product, a dot b equals to A b. 
        # this function makes a to A 
        x = vector3d[0]
        y = vector3d[1]
        z = vector3d[2]

        skew_sym_mat = torch.zeros((3,3), dtype=torch.float32).to("cuda:1")
        skew_sym_mat[0,1] = -z
        skew_sym_mat[0,2] = y
        skew_sym_mat[1,0] = z
        skew_sym_mat[1,2] = -x
        skew_sym_mat[2,0] = -y
        skew_sym_mat[2,1] = x

        return skew_sym_mat
    

    R = pose_mat[:3,:3]
    t = pose_mat[:3, 3]
    E_mat = veccrossproduct(t)@ R
    F_mat = torch.linalg.inv(intrinsics).transpose(1,0) @ E_mat @ torch.linalg.inv(intrinsics)
    
    # to_be_zero = kpts_2 @ F_mat @ kpts_1.transpose(1,0)
    kpts_2 = HomographicSampler._convert_points_to_homogeneous(kpts_2)
    kpts_1 = HomographicSampler._convert_points_to_homogeneous(kpts_1)
    
    F_mat = F_mat.repeat(kpts_1.shape[0], 1, 1)
    # print(F_mat.shape)    torch.Size([10001, 3, 3])

    to_be_zero = torch.bmm(F_mat, kpts_1.unsqueeze(2))
    to_be_zero = torch.bmm(kpts_2.unsqueeze(1), to_be_zero)

    # print(to_be_zero.shape)torch.Size([10001, 1, 1])
    to_be_zero = to_be_zero.squeeze().abs().sum()
    print("--------------------------------")
    print(to_be_zero)
    
    return to_be_zero



def ones_like_loss(probability):
    print(probability.shape)
    ones = torch.ones_like(probability) #.to(probability.device)  
    loss = F.binary_cross_entropy(probability, ones)

    return loss 


    
def patch_positions(img, kpts, patch_size:int = 8):
    
    # deep copy keypoints
    kpts_ = kpts.clone()
    if kpts_.shape[0]==0:
        print(kpts_.shape)
        print("mesh is empty")
        exit(0)
    
    
    # kpts: row, column order
    # crop point: row,column order
    # kpts_[:,0] += crop_point[0].to(kpts_.device)
    # kpts_[:,1] += crop_point[1].to(kpts_.device)
    # kpts_[:,0] += crop_point[0]
    # kpts_[:,1] += crop_point[1]

        
    # below this line assumes x==column & y==row order
    ys = []
    xs = []
    for i in range(patch_size): #i : 0, 1, 2, 3, 4, ..., 7
        ys.append(kpts[:,0] - (patch_size - i))
        xs.append(kpts[:,1] - (patch_size - i))
    for i in range(patch_size): #i : 0, 1, 2, 3, 4, ..., 7
        ys.append(kpts[:,0] + i+1)
        xs.append(kpts[:,1] + i+1)
        
    ys = torch.stack(ys, dim=-1)
    xs = torch.stack(xs, dim=-1)
    
    
    # True
    mesh = []
    for i in range(kpts_.shape[0]): # i = 0 ~ 3000
        mesh.append(torch.stack(torch.meshgrid([xs[i], ys[i]], indexing="ij"), dim=-1).reshape(-1,2))
    # print(mesh[0].shape) torch.Size([256, 2])
    toreturn = torch.stack(mesh, dim = 0).reshape(-1,2) # (patch_area x num_kpt) x 2 torch.Size([3001, 256, 2])

    # torch.Size([768256, 2])
    
    print(img.shape)
    # torch.Size([1, 3, 370, 1226])

    toreturn[:, 0] = 2*toreturn[:, 0]/(img.shape[-1]) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    toreturn[:, 1] = 2*toreturn[:, 1]/(img.shape[-2]) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]    
    
    
    return toreturn.reshape(1, kpts.shape[0], -1, 2)


def photometric_reconstruction_loss(whole_images, depth_map_0, depth_map_1, intrinsics, pose_gt_forward, pose_gt_backward, logits, coord_mapping=None):
    
    # io.imsave("./folder_for_viz/whole_image_1.png", whole_images[0].squeeze(0).permute(1,2,0).detach().cpu().numpy())
    # io.imsave("./folder_for_viz/whole_image_2.png", whole_images[1].squeeze(0).permute(1,2,0).detach().cpu().numpy())
    
    # im_1, im_2: 0~255 with 3 channel    
    recon_loss = 0
    des_loss = 0
    shape = whole_images.shape
    im_0, im_1 = whole_images[0].unsqueeze(0), whole_images[1].unsqueeze(0)
    
    if len(depth_map_1.shape) == 2:
        depth_map_0, depth_map_1 = depth_map_0.unsqueeze(0), depth_map_1.unsqueeze(0)
    
    positions = HomographicSampler._create_meshgrid(
        im_1.shape[-2],
        im_1.shape[-1],
        device=im_1.device,
        normalized=False,
    )
    positions = HomographicSampler._convert_points_to_homogeneous(positions.reshape(1, -1, 2)).permute(0,2,1)
    
    im_0 = im_0.to(torch.float32)
    im_1 = im_1.to(torch.float32)
        
    # print(logits.shape, positions.shape)
    # torch.Size([2, 1, 352, 1208]) torch.Size([1, 3, 453620])

    # pose 1->2 
    recon_loss += compute_diff(intrinsics, pose_gt_backward, logits[0], im_1.clone(), im_0.clone(), depth_map_0, positions.clone(), shape)
    recon_loss += compute_diff(intrinsics, pose_gt_forward, logits[1], im_0.clone(), im_1.clone(), depth_map_1, positions.clone(), shape)
    
    return recon_loss



def compute_diff(intrinsics, pose_inv, logits_1, image_0, image_1, depth_map_1, positions, shape):

    image_0_warped, valid_points_1, bf_norm = inverse_warp(image_0, depth_map_1, pose_inv,
                                                intrinsics, positions.clone(), shape)
    # io.imsave("folder_for_viz/im_0_warped.png", (255*image_0_warped[0]).permute(1,2,0).detach().cpu().numpy())
    # io.imsave("folder_for_viz/im_0.png", (255*image_0[0]).permute(1,2,0).detach().cpu().numpy())
   
    # print(image_1.shape, image_0_warped.shape)
    # print(max(image_1.reshape(-1)), max(image_0_warped.reshape(-1)))
    # print(min(image_1.reshape(-1)), min(image_0_warped.reshape(-1)))
    
    diff = abs(image_1*valid_points_1 - image_0_warped)
    # print(diff.shape)
    # torch.Size([1, 1, 370, 1226])
    # print(logits_1.shape)
    # torch.Size([1, 352, 1208])
    # print(diff[0, :, 9:-9, 9:-9].shape)
    # torch.Size([1, 352, 1208])

    photo_loss = logits_1 * diff[0, :, 9:-9, 9:-9]
    # io.imsave("folder_for_viz/diff.png", (255*diff[0]).permute(1,2,0).detach().cpu().numpy())
    # io.imsave("folder_for_viz/diff_weighted.png", (255*photo_loss).permute(1,2,0).detach().cpu().numpy())
    

    return photo_loss.mean()



def convert_points_from_homogeneous(
        points: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        # we check for points at max_val
        z_vec: torch.Tensor = points[..., -1:]

        # set the results of division by zeror/near-zero to 1.0
        # follow the convention of opencv:
        # https://github.com/opencv/opencv/pull/14411/files
        mask: torch.Tensor = torch.abs(z_vec) > eps
        scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

        return scale * points[..., :-1]


def inverse_warp(img, depth, pose_mat, intrinsics, positions, shape, padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    # """
    # print("=========")
    # print(depth.shape)
    # torch.Size([1, 370, 1226])
    # torch.Size([1, 370, 1226])
    # io.imsave("./folder_for_viz/depth.png", depth.squeeze(0).cpu().numpy())
    # ValueError: Image must be 2D (grayscale, RGB, or RGBA).
    
    # print(positions.shape)
    # =========
    # torch.Size([1, 3, 453620])
    # torch.Size([1, 3, 453620])
    # print(max(positions[0,0,:]))
    # print(max(positions[0,1,:]))
    # print(max(positions[0,2,:]))

    cam_coords = HomographicSampler.pixel2cam(depth, intrinsics.inverse(), positions, shape)  # [B,3,H,W]
    # print(max(cam_coords[0,0,:]))
    # print(max(cam_coords[0,1,:]))
    # print(max(cam_coords[0,2,:]))
    # tensor(0.2530, device='cuda:1')
    # tensor(0.0391, device='cuda:1')
    # tensor(1., device='cuda:1')

    # print(cam_coords.shape)
    # torch.Size([1, 3, 53248])
    # torch.Size([1, 3, 453620])
    
    pose_mat = pose_mat[:3,:4].unsqueeze(0).double()
    # print(shape)
    # (tensor([3]), tensor([370]), tensor([1226]))

    src_pixel_coords, bf_norm  = HomographicSampler.cam2pixel_forward(cam_coords.double(), pose_mat, intrinsics.double(), normalize=True, shape = shape)
    # print(max(src_pixel_coords[0,:,0]))
    # print(max(src_pixel_coords[0,:,1]))
    # tensor(0.9921, device='cuda:1')
    # tensor(0.9356, device='cuda:1')
    
    # tensor(286.0254, device='cuda:1')
    # tensor(47.5957, device='cuda:1')
    # this is wrong

    # print(src_pixel_coords.shape)
    # torch.Size([1, 53248, 2])
    if len(src_pixel_coords.shape)==3:
        src_pixel_coords = src_pixel_coords.reshape(1, img.shape[-2], img.shape[-1], 2)
        bf_norm = bf_norm.reshape(1, img.shape[-2], img.shape[-1], 2)
    # print(img.shape, src_pixel_coords.shape)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    # print(depth.shape)
    # print(img.shape)
    # print(src_pixel_coords.shape)
    # torch.Size([1, 370, 1226])
    # torch.Size([1, 3, 370, 1226])
    # torch.Size([1, 370, 1226, 2])

    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
    # print(projected_img.shape)
    # torch.Size([1, 3, 370, 1226])

    # io.imsave("./folder_for_viz/img.png", img.squeeze(0).permute(1,2,0).detach().cpu().numpy())
    # io.imsave("./folder_for_viz/projected_img.png", projected_img.squeeze(0).permute(1,2,0).detach().cpu().numpy())

    # torch.Size([1, 128, 416, 2])

    # src_pixel_coords = torch.tensor([[[-1.2, -0.9],[0.2, 0.3],[1.0, 1.1]],
    #                                  [[-0.8, -0.8],[0.0, 0.1],[0.9, 1.0]],
    #                                  [[-1.4, -0.9],[0.0, -0.3],[1.2, 1.0]]])
    # print(src_pixel_coords.shape)
    # torch.Size([3, 3, 2])

    valid_points = torch.where(src_pixel_coords.abs().max(dim=-1)[0]<=1, True, False)
    # print(valid_points.shape)
    # torch.Size([1, 128, 416])

    # print(torch.count_nonzero(valid_points))
    # tensor(49327, device='cuda:1')
    
    return projected_img, valid_points, bf_norm


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    # print("in_euler2mat", angle)
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
    
    # print(x, y, z)
    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z*0
    ones = zeros+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1]*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:] 
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


