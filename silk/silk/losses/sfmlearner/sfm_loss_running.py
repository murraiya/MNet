import numpy as np
import torch
import skimage.io as io
import torch.nn.functional as F
from silk.matching.mnn import mutual_nearest_neighbor
from silk.cv.homography import HomographicSampler
# from silk.cv.homography_warp_bidirectional import HomographicSampler


# self.flow.define_transition(
#             ("pose_loss", "matched_sparse_positions_1", "matched_sparse_positions_2", "matches"),
#             epiploar_loss,
#             intrinsics,
#             pose_gt_forward,
#             "sparse_positions_1",
#             "sparse_positions_2",
#             "sparse_descriptors",
#         )

def epiploar_loss(intrinsics, pose_mat, kpts_1_, kpts_2_, descriptors):
    # kpts in pixel coord needed
    # pred or gt no matters
    # print("+++++++++++++++++")
    # print(kpts_1.shape)
    # torch.Size([10001, 3])
    # print(len(descriptors), type(descriptors))
    # print(descriptors[0].shape, descriptors[1].shape)
    # 2 <class 'torch.Tensor'>
    # torch.Size([128, 110, 398]) torch.Size([128, 110, 398])
    
    print("what the matches do")
    # print(kpts_1.shape, kpts_2.shape)
    # print(descriptors[0].shape, descriptors[1].shape)
    # they should match 
    matches = mutual_nearest_neighbor(
        descriptors[0],
        descriptors[1]
    )
    # print("epip_loss")
    # print(pose_mat.requires_grad)
    # print(intrinsics.requires_grad)
    # print(descriptors[0].requires_grad)
    # print(matches.requires_grad)
    # print(kpts_1.requires_grad)
    # False
    # False
    # True
    # False
    # True
    kpts_1 = kpts_1_.clone()
    kpts_2 = kpts_2_.clone()

    # print(matches.shape)
    kpts_2 = (kpts_2[matches[:, 1]])[:, [1,0]]
    kpts_1 = (kpts_1[matches[:, 0]])[:, [1,0]]
    # print(kpts_1.shape, kpts_2.shape)
    # what the matches do
    # torch.Size([10001, 3]) torch.Size([10001, 3])
    # torch.Size([4034, 2])
    # torch.Size([4034, 2]) torch.Size([4034, 2])

        
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
    
    # print(pose_mat.shape)
    # torch.Size([4, 4])
    R = pose_mat[:3,:3]
    t = pose_mat[:3, 3]
    # print(type(R), type(t), type(pose_mat))
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


# print(max(xgrid), max(ygrid))
# xgrid = 2*xgrid/(W-1) - 1
# ygrid = 2*ygrid/(H-1) - 1

# grid = torch.cat([xgrid, ygrid], dim=-1).unsqueeze(0).unsqueeze(1)
# mask = torch.ones(image_1.size()).to("cuda:1")
# mask = F.grid_sample(mask, grid)


# intrinsics,
# pose_gt_forward, 
# pose_gt_backward,
# "matched_sparse_positions_1",
# "matched_sparse_positions_2",
# "matches",
# "gray_images",
# depth_map_1,
# depth_map_2,

# def patchy(img, kpts, patch_size = 8):
#     img_height = img.shape[-2]
#     img_width = img.shape[-1]
#     # print(img_height, img_width) 128 416

#     patch_mask = torch.full((img.shape[-2:]), 0, device=img.device)
#     # print("patch_mask.shape", patch_mask.shape)
#     # patch_mask.shape torch.Size([128, 416])

#     # print(kpts.shape)
#     # torch.Size([1455, 3])
#     # print(kpts.shape)
#     # print(min(kpts[:,0]), max(kpts[:,0]))
#     # print(min(kpts[:,1]), max(kpts[:,1]))
#     # torch.Size([3001, 3])
#     # tensor(9.5000, device='cuda:1') tensor(118.5000, device='cuda:1')
#     # tensor(9.5000, device='cuda:1') tensor(406.5000, device='cuda:1')
#     print("patchy")
#     print(kpts.requires_grad)
#     # True

#     kpts = HomographicSampler._convert_points_from_homogeneous(kpts)
    
#     print(kpts.requires_grad)
#     # True
#     # kpts = kpts[:, [1,0]] # change to x,y order
    
#     # print(kpts.shape)torch.Size([3001, 2])
#     # print(min(kpts[:,0]), max(kpts[:,0]))
#     # print(min(kpts[:,1]), max(kpts[:,1]))
#     # tensor(9.5000, device='cuda:1') tensor(356.5000, device='cuda:1')
#     # tensor(9.5000, device='cuda:1') tensor(117.5000, device='cuda:1')


#     # below this line assumes x==column & y==row order
#     xs_l = kpts[:,1] - patch_size
#     xs_r = kpts[:,1] + patch_size+1
#     ys_u = kpts[:,0] - patch_size
#     ys_b = kpts[:,0] + patch_size+1
    
#     print(xs_l.requires_grad)
#     # True
    
#     # clamping
#     xs_l = torch.where(xs_l>=0,          xs_l, 0         ).to(torch.int32)
#     xs_r = torch.where(xs_r<=img_width,  xs_r, img_width ).to(torch.int32)
#     ys_u = torch.where(ys_u>=0,          ys_u, 0         ).to(torch.int32)
#     ys_b = torch.where(ys_b<=img_height, ys_b, img_height).to(torch.int32)
#     print(xs_l.requires_grad)
#     # True
#     # xs_l = xs_l.to(torch.int32)
#     # print(xs_l.requires_grad)
#     # False

    
#     # print(min(xs_l), max(xs_l))
#     # print(min(xs_r), max(xs_r))
#     # print(min(ys_u), max(ys_u))
#     # print(min(ys_b), max(ys_b))
#     # tensor(6, device='cuda:1', dtype=torch.int32) tensor(348, device='cuda:1', dtype=torch.int32)
#     # tensor(13, device='cuda:1', dtype=torch.int32) tensor(355, device='cuda:1', dtype=torch.int32)
#     # tensor(6, device='cuda:1', dtype=torch.int32) tensor(114, device='cuda:1', dtype=torch.int32)
#     # tensor(13, device='cuda:1', dtype=torch.int32) tensor(121, device='cuda:1', dtype=torch.int32)

#     # patches = torch.stack([xs_l,ys_u,xs_r, ys_b], dim=1) #made nx4, n pathces

#     for i in range(kpts.shape[0]):
#         patch_mask[ys_u[i]:ys_b[i], torch.round(xs_l[i]):xs_r[i]] = 1
        
#     print(patch_mask.requires_grad)
#     # False
    
#     # print("patch mask")
#     # print(patch_mask.shape)
#     # print(torch.count_nonzero(patch_mask), kpts.shape) 
#     # tensor(5449) torch.Size([1806, 2])

    
#     return patch_mask    

# working
# def patch_positions(img, kpts, patch_size = 8):
#     print("patch_positions")
#     print(kpts.requires_grad)
#     # True

#     # below this line assumes x==column & y==row order
#     xs_l = kpts[:,0] - patch_size
#     xs_r = kpts[:,0] + patch_size -1
#     ys_u = kpts[:,1] - patch_size
#     ys_b = kpts[:,1] + patch_size -1
    
#     print(kpts.requires_grad)
#     # True
#     mesh = []
#     for i in range(kpts.shape[0]):
#         xs: torch.Tensor = torch.linspace(
#             xs_l[i],
#             xs_r[i],
#             2*patch_size,
#             device=kpts.device,
#             dtype=kpts.dtype,
#         )
#         ys: torch.Tensor = torch.linspace(
#             ys_u[i],
#             ys_b[i],
#             2*patch_size,
#             device=kpts.device,
#             dtype=kpts.dtype,
#         )
        
#         # meshgrid returns tuple
#         mesh.append(torch.stack(torch.meshgrid([xs, ys], indexing="ij"), dim=-1).reshape(-1,2))
#     print(mesh.requires_grad)
#     # False
#     print("here")

#     print(len(mesh))
#     # 3001
#     print(mesh[0].shape)
#     # torch.Size([256, 2])

#     toreturn = torch.stack(mesh, dim = 0).reshape(-1,2) # (patch_area x num_kpt) x 2 torch.Size([3001, 256, 2])
#     toreturn[:, 0] = 2*toreturn[:, 0]/(img.shape[-1]) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
#     toreturn[:, 1] = 2*toreturn[:, 1]/(img.shape[-2]) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]    
    
#     print(kpts.requires_grad)
#     print("++++++++")
#     return toreturn.reshape(1, kpts.shape[0], -1, 2)



    
def patch_positions(img, kpts, patch_size:int = 8):
    print("patch_positions")
    print(kpts.requires_grad)
    # True

    
    # deep copy keypoints
    kpts_ = kpts.clone()
    # print(kpts.shape)
    # torch.Size([3001, 3])

    # below this line assumes x==column & y==row order
    ys = []
    xs = []
    for i in range(patch_size): #i : 0, 1, 2, 3, 4, ..., 7
        ys.append(kpts[:,0] - (patch_size - i))
        xs.append(kpts[:,1] - (patch_size - i))
    for i in range(patch_size): #i : 0, 1, 2, 3, 4, ..., 7
        ys.append(kpts[:,0] + i+1)
        xs.append(kpts[:,1] + i+1)
        
    print(len(xs)) #16 16x3001x1
    ys = torch.stack(ys, dim=-1)
    xs = torch.stack(xs, dim=-1)
    print(xs.shape) #torch.Size([3001, 16])
    print(xs)
    # tensor([[ 21.5000,  22.5000,  23.5000,  ...,  35.5000,  36.5000,  37.5000],
    #         [ 28.5000,  29.5000,  30.5000,  ...,  42.5000,  43.5000,  44.5000],
    #         [ 29.5000,  30.5000,  31.5000,  ...,  43.5000,  44.5000,  45.5000],
    #         ...,
    #         [284.5000, 285.5000, 286.5000,  ..., 298.5000, 299.5000, 300.5000],
    #         [270.5000, 271.5000, 272.5000,  ..., 284.5000, 285.5000, 286.5000],
    #         [284.5000, 285.5000, 286.5000,  ..., 298.5000, 299.5000, 300.5000]],
    #     device='cuda:1')

    
    
    print(xs.requires_grad)
    print(kpts.shape)
    print(kpts_.shape)
    # torch.Size([3001, 3])
    # torch.Size([3001, 3])
    
    # True
    mesh = []
    for i in range(kpts_.shape[0]): # i = 0 ~ 3000
        mesh.append(torch.stack(torch.meshgrid([xs[i], ys[i]], indexing="ij"), dim=-1).reshape(-1,2))
    print(len(mesh)) #3001
    # print(mesh[0].shape) torch.Size([256, 2])

    toreturn = torch.stack(mesh, dim = 0).reshape(-1,2) # (patch_area x num_kpt) x 2 torch.Size([3001, 256, 2])

    print(toreturn.shape)
    # torch.Size([768256, 2])
    
    
    toreturn[:, 0] = 2*toreturn[:, 0]/(img.shape[-1]) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    toreturn[:, 1] = 2*toreturn[:, 1]/(img.shape[-2]) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]    
    
    print(kpts.requires_grad)
    print(toreturn.requires_grad)
    
    print("++++++++")
    return toreturn.reshape(1, kpts.shape[0], -1, 2)


def photometric_reconstruction_loss(intrinsics, pose, pose_inv, kpts_1, kpts_2, images, depth_map_1, depth_map_2):
    print("photometric")
    print(intrinsics.requires_grad, pose.requires_grad, pose_inv.requires_grad)
    print(kpts_1.requires_grad, kpts_2.requires_grad)
    print(images.requires_grad, depth_map_1.requires_grad, depth_map_2.requires_grad)
    # False False False
    # True True
    # False False False
    
    reconstruction_loss = 0
    image_1, image_2 = 255*images[0], 255*images[1]
    if len(image_1.shape) == 3:
        image_1 = image_1.unsqueeze(0)
        image_2 = image_2.unsqueeze(0)

    # descriptors_height = descriptors.shape[2]
    # descriptors_width = descriptors.shape[3]
    # print(depth_map_1.shape, depth_map_2.shape)    
    # torch.Size([128, 416]) torch.Size([128, 416]) torch.Size([1456, 2])

    if len(depth_map_1.shape) == 2:
        depth_map_1, depth_map_2 = depth_map_1.unsqueeze(0), depth_map_2.unsqueeze(0)
    # print(image_1.shape)torch.Size([1, 128, 416])
    
    positions = HomographicSampler._create_meshgrid(
        image_1.shape[-2],
        image_1.shape[-1],
        device=image_1.device,
        normalized=False,
    )
    
    positions = HomographicSampler._convert_points_to_homogeneous(positions.reshape(1, -1, 2)).permute(0,2,1)
    print(positions.shape)
    # torch.Size([1, 3, 53248])

    # print(max(positions[0,0,:]), max(positions[0,1,:]))
    # tensor(415.5000, device='cuda:1') tensor(127.5000, device='cuda:1')
    
    
    
    print(kpts_1.shape)
    # torch.Size([3001, 3])
    # print(min(kpts_1[:,0]), max(kpts_1[:,0]))
    # print(min(kpts_1[:,1]), max(kpts_1[:,1]))
    # print(min(kpts_1[:,2]), max(kpts_1[:,2])) # this is confidence value. dont think as z
    # tensor(9.5000, device='cuda:1') tensor(118.5000, device='cuda:1')
    # tensor(9.5000, device='cuda:1') tensor(406.5000, device='cuda:1')
    # tensor(0.6950, device='cuda:1') tensor(0.9789, device='cuda:1')
# print(img.shape, src_pixel_coords.shape)
        
    reconstruction_loss += compute_diff(intrinsics, pose_inv, kpts_2.clone(), image_1, image_2, depth_map_2, positions.clone())
    reconstruction_loss += compute_diff(intrinsics, pose,     kpts_1.clone(), image_2, image_1, depth_map_1, positions.clone())

    print("----------------!!!!!!!!!!!!!----------------")
    print(reconstruction_loss)
    return reconstruction_loss
    
    
#im_1

def compute_diff(intrinsics, pose_inv, kpts_2, image_1, image_2, depth_map_2, positions):
    # positions input to inversewarp should be b, 3, H*W
    # mask = patch_positions(image_1, kpts_1)
    image_1_warped, valid_points_1 = inverse_warp(image_1, depth_map_2, pose_inv,
                                                intrinsics, positions.clone())
    #image_1_warped should be same with image_2 ideally
    # diff = (image_1_corr_img - image_2_warped) * valid_points.unsqueeze(1).float()
    # mask = patchy(image_2, kpts_2)
    # print("mask")
    # print(mask.requires_grad)

    # print(image_1_warped.shape, image_2.shape, mask.shape, valid_points.shape)
    # torch.Size([1, 128, 416]) torch.Size([1, 128, 416]) torch.Size([128, 416]) torch.Size([1, 128, 416])
    
    print(kpts_2.requires_grad)
    kpts2_positions = patch_positions(image_2, kpts_2.clone())
    # kpts2_positions = torch.gather(image_2, kpt2)
    print(kpts2_positions.shape, image_2.shape)
    # torch.Size([1, 3001, 256, 2]) torch.Size([1, 1, 128, 416])

    image_2_corresponding = F.grid_sample(image_2, kpts2_positions, padding_mode="zeros", align_corners=False)
    
    
    image_1_corresponding = F.grid_sample(image_1_warped, kpts2_positions, padding_mode="zeros", align_corners=False)
    valid_points_2 = torch.where(kpts2_positions.abs().max(dim=-1)[0]<=1, True, False)
    
    print(image_1_corresponding.shape, image_2_corresponding.shape)
    # torch.Size([1, 1, 3001, 256]) torch.Size([1, 1, 3001, 256])

    diff = abs(image_2_corresponding - image_1_corresponding)      
    print(diff.shape)
    # torch.Size([1, 1, 3001, 256])

    
    # io.imsave("./folder_for_viz/mask.png", mask.unsqueeze(2).cpu().numpy())
    # io.imsave("./folder_for_viz/valid.png", valid_points.permute(1,2,0).cpu().numpy())
    
    # io.imsave("./folder_for_viz/corres_im_2.png", image_2_corresponding.squeeze(0).reshape(1,256,-1).permute(1,2,0).detach().cpu().numpy())
    # io.imsave("./folder_for_viz/corres_im_1.png", image_1_corresponding.squeeze(0).reshape(1,256,-1).permute(1,2,0).detach().cpu().numpy())
    return diff.mean()





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




# image_1_warped, valid_points = inverse_warp(image_1, depth_map_2, pose_inv,
#                                                 intrinsics, positions)
def inverse_warp(img, depth, pose_mat, intrinsics, positions, padding_mode='zeros', shape=None):
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
    
    cam_coords = HomographicSampler.pixel2cam(depth, intrinsics.inverse(), positions, depth.shape)  # [B,3,H,W]
    # print(cam_coords.shape)
    # torch.Size([1, 3, 53248])
    pose_mat = pose_mat[:3,:4].unsqueeze(0).double()
    src_pixel_coords = HomographicSampler.cam2pixel_forward(cam_coords.double(), pose_mat, intrinsics.double(), normalize=True)
    # print(src_pixel_coords.shape)
    # torch.Size([1, 53248, 2])
    if len(src_pixel_coords.shape)==3:
        src_pixel_coords = src_pixel_coords.reshape(1, img.shape[-2], img.shape[-1], 2)

    # print(img.shape, src_pixel_coords.shape)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
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
    
    return projected_img, valid_points



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


