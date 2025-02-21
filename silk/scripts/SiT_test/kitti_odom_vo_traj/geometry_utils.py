import numpy as np
import torch

def pose_loss(gt, pred):
    pred = pred[:1, :, :] # pose6d of non-warped image, should be (1, 1, 6)
    # print("in pose_loss: ", gt, pred)
    # print("in pose_loss: ", gt.shape, pred.shape)
    # in pose_loss:  [[[ 9.99816833e-01  4.54430856e-03 -1.85927258e-02 -4.05646605e-01]
    # [-4.73810611e-03  9.99934698e-01 -1.03925531e-02 -2.46380486e-01]
    # [ 1.85442855e-02  1.04787480e-02  9.99773150e-01  7.72423898e+00]
    # [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]] tensor([[[ 0.0005,  0.0002, -0.0005,  0.0004,  0.0004,  0.0008]]],
    #     device='cuda:1')
    # in pose_loss:  (1, 4, 4) torch.Size([1, 1, 6])

    pred = pred[0]
    inv_transform_matrices = pose_vec2mat(pred, rotation_mode="euler")
    gt = torch.from_numpy(gt).to("cuda:1")
    # print("in pose_loss: ", inv_transform_matrices.shape)
    # in pose_loss:  (1, 3, 4)

    # rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
    # tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

    # transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)
    # print("in pose_loss: ", transform_matrices.shape)

    # first_inv_transform = inv_transform_matrices[0]
    # final_poses = first_inv_transform[:,:3] @ transform_matrices
    # final_poses[:,:,-1:] += first_inv_transform[:,-1:]
    
    l2loss = compute_pose_loss(gt[:,:-1,:], inv_transform_matrices) # make gt 1x3x4
    return l2loss

def compute_pose_loss(gt, pred):
    # gt.reshape(12)
    # pred.rehape(12)
    # print(gt.reshape(-1).shape)
    # print(gt.reshape(-1))
    return torch.linalg.norm(gt.reshape(-1)-pred.reshape(-1))


def euler2mat(angle):
    """Convert euler angles to rotation matrix.

     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
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
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
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



# if __name__ == '__main__':
    # ATE, RE = compute_pose_error(sample['poses'], final_poses)

    # poses = poses.cpu()[0]
    # poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])

    # inv_transform_matrices = pose_vec2mat(poses, rotation_mode=args.rotation_mode).numpy().astype(np.float64)

    # rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
    # tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

    # transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

    # first_inv_transform = inv_transform_matrices[0]
    # final_poses = first_inv_transform[:,:3] @ transform_matrices
    # final_poses[:,:,-1:] += first_inv_transform[:,-1:]

    # if args.output_dir is not None:
    #     predictions_array[j] = final_poses

    # ATE, RE = compute_pose_error(sample['poses'], final_poses)