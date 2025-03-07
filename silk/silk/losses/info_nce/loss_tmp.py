# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import math
from silk.matching.mnn import (
    compute_dist
)
# import numpy as np

def positions_to_unidirectional_correspondence(
    positions,
    width,
    height,
    cell_size,
    ordering="yx",
):
    assert ordering in {"xy", "yx"}

    # positions : Nx2
    # cell_size = torch.tensor(cell_size, dtype=positions.dtype)
    # cell_size = cell_size.unsqueeze(0, 1)

    # floored_positions = torch.floor(positions / cell_size).astype(torch.int32)
    device = positions.device
    # print(positions.shape)
    # # torch.Size([1, 43780, 2])
    # print(min(positions[0,:,0]), max(positions[0,:,0]))
    # print(min(positions[0,:,1]), max(positions[0,:,1]))

    floored_positions = torch.floor(positions).to(torch.int32)
    # a = floored_positions.clone()
    # print(floored_positions.shape)
    # torch.Size([1, 43780, 2])
    # print(min(floored_positions[0,:,0]), max(floored_positions[0,:,0]))
    # print(min(floored_positions[0,:,1]), max(floored_positions[0,:,1]))
    # tensor(0, device='cuda:1', dtype=torch.int32) tensor(379, device='cuda:1', dtype=torch.int32)
    # tensor(-1, device='cuda:1', dtype=torch.int32) tensor(103, device='cuda:1', dtype=torch.int32)


    if ordering == "yx":
        desc_shape = torch.tensor([[height, width]]).to(device)
    elif ordering == "xy":
        desc_shape = torch.tensor([[width, height]]).to(device)

    mask = torch.logical_and(floored_positions >= 0, floored_positions < desc_shape)
    mask = mask.all(axis=2)

    if ordering == "yx":
        floored_positions = (
            floored_positions[..., 0] * width + floored_positions[..., 1]
        )
    elif ordering == "xy":
        floored_positions = (
            floored_positions[..., 1] * width + floored_positions[..., 0]
        )
    # print(floored_positions.shape)
    # print(mask.shape)
    # # torch.Size([1, 43780])
    # # torch.Size([1, 43780])

    floored_positions = torch.where(mask, floored_positions, -1)
    
    return floored_positions



def sparse_positions_to_corr(sparse_positions_0, wapred_positions_1):

    # print(sparse_positions_0.shape, wapred_positions_1.shape)
    # torch.Size([1, 7001]) torch.Size([1, 7001])

    idx = torch.arange(sparse_positions_0.shape[1], device=wapred_positions_1.device)
    # print(idx)
    is_bidir = sparse_positions_0[0] == wapred_positions_1[0]
    return torch.where(is_bidir, idx, -1).unsqueeze(0)


def asym_keep_mutual_correspondences_only(corr_0_, corr_1_):
    # print(corr_1_.shape)
    # print(corr_0_.shape)
    # torch.Size([1, 43780])
    # torch.Size([1, 43780])

    corr_0 = corr_0_.clone()[0]
    corr_1 = corr_1_.clone()[0]
    idx = torch.arange(corr_0.shape[0], device=corr_0.device)
   
    is_bidir = corr_1[corr_0] == idx
    return torch.where(is_bidir, corr_0, -1).unsqueeze(0)


def keep_mutual_correspondences_only(corr_0, corr_1):
    corr_0 = asym_keep_mutual_correspondences_only(corr_0, corr_1)
    corr_1 = asym_keep_mutual_correspondences_only(corr_1, corr_0)

    return corr_0, corr_1


def _scan_reduce(x0, x1, reducer, block_size):
    
    x0 = x0.squeeze(0)
    x1 = x1.squeeze(0)
    
    x0_shape0 = x0.shape[0]
    n = x0.shape[0] // block_size
    
    if x0.shape[0] % block_size > 0:

        r = block_size - x0.shape[0] % block_size
        _0 = torch.tensor(0, dtype=x0.dtype)
        x0 = torch.nn.functional.pad(x0, (0, 0, 0, r), "constant", _0)
        n += 1
        # get rid of paded parts when return 

    x0 = x0.reshape(n, block_size, x0.shape[1])
    xs = x0.clone()
    
    def scan(f, init, xs, length=None):
        if xs is None:
            xs = [None] * length
        carry = init #None
        ys = []
        
        a = 0
        for x in xs:
            a+=1

            carry, y = f(carry, x)  # carry is the carryover
            ys.append(y)            # the `y`s get accumulated into a stacked array
        
        return carry, torch.stack(ys, dim=1)
    
    
    def fun(_, x0):
        return None, reducer(x0, x1)
    
    _, accu = scan(fun, None, xs, length=n)
    
    
    return torch.ravel(accu[0])[:x0_shape0], torch.ravel(accu[1])[:x0_shape0], torch.ravel(accu[2])[:x0_shape0]
    






    # ## i prefer small difference (dummy score) 
    # descs_1 = torch.where(dummy_score<0.73, descs_1, 0)
    # # a = torch.nonzero(dummy_score>avg)
    # # print(descs_1[a[1]]) confirmed. it is all 0
    
    # # should check if this cuts gradient 
    # # warn !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # # print(a.shape)
    # # torch.Size([128, 370, 1226])
    # # a = a.sum(dim=0)
    # # print(a.shape)
    # # torch.Size([370, 1226])
    # # print(torch.count_nonzero(a))
    # # tensor(172107, device='cuda:1')
    # # ------------------------------------------------ til here, handled only 1
    
    # # take corresponding desc_0 using bf_norm 
    # # bf norm is coordinate of image_0 corresponding to every coord in image_1
    # # torch.Size([1, 370, 1226, 2])
    
    # # print(max(bf_norm.reshape(-1,2)[:,0]), max(bf_norm.reshape(-1,2)[:,1]))
    # # tensor(1364.3596, device='cuda:1', dtype=torch.float64) tensor(369.5000, device='cuda:1', dtype=torch.float64)

    # mask = positions_to_unidirectional_correspondence(
    #     bf_norm.reshape(1,-1,2), 
    #     width=logits_0.shape[2], 
    #     height=logits_0.shape[1], 
    #     cell_size = 1.0,
    #     ordering="xy"
    # )

    # # descs_0 = descs_0.reshape(descs_0.shape[0], -1)
    # # # print(descs_0.shape, mask.shape)
    # # # torch.Size([128, 453620]) torch.Size([1, 453620])
    # # descs_0 = torch.where(mask>-1, descs_0[mask], 0)
    # # # cuda oom 

    # mask_ = mask.repeat(descs_0.shape[0], 1).transpose(1,0)
    # # print(mask.shape)
    # descs_0 = descs_0.reshape(descs_0.shape[0], -1).transpose(1,0)
    
    # # print(descs_0.shape, mask.shape)
    # # torch.Size([453620, 128]) torch.Size([1, 453620])

    
    # descs_0 = torch.where(mask_>-1, descs_0[mask], 0)
    # # print(descs_0.shape)
    # # torch.Size([1, 453620, 128])

    # # print(_desc_1.shape)
    # # torch.Size([43780, 128])
    
    
    # # print(descs_0.shape, descs_1.shape)
    # # torch.Size([453620, 128]) torch.Size([128, 370, 1226])

    # descs_1 = descs_1.reshape(descs_1.shape[0], -1).transpose(1,0)
    # # aligned dot product
    # # log_num = torch.bmm(descs_0.squeeze(0).unsqueeze(1), descs_1.unsqueeze(2)).reshape(-1,1)
    # # print(log_num.shape)
    # # torch.Size([453620, 1])
    # # print(descs_0.shape, descs_1.shape)
    # # torch.Size([1, 453620, 128]) torch.Size([453620, 128])
    # cos_sim = F.cosine_similarity(descs_0[0], descs_1)
    # # print(cos_sim.shape)
    # # torch.Size([453620])
    # # print(max(cos_sim), min(cos_sim))
    # # tensor(1.0000, device='cuda:1') tensor(0., device='cuda:1')

    # ones_var = torch.ones_like(cos_sim)
    # cos_dist = ones_var - cos_sim
    # cos_dist = cos_dist.mean()    

    # # io.imsave("./folder_for_viz/pooled_32.png", a.squeeze(0).permute(1,2,0).detach().cpu().numpy())

    # # io.imsave("./folder_for_viz/mask.png", mask.unsqueeze(2).cpu().numpy())
    # # io.imsave("./folder_for_viz/valid.png", valid_points.permute(1,2,0).cpu().numpy())
    
    # # io.imsave("./folder_for_viz/corres_im_2.png", image_2_corresponding.squeeze(0).permute(1,2,0).detach().cpu().numpy())
    # # io.imsave("./folder_for_viz/corres_im_1.png", image_1_corresponding.squeeze(0).permute(1,2,0).detach().cpu().numpy())
    
    # # print(type(photo_loss))
    # return photo_loss, cos_dist


def asym_corr_cross_entropy(
    lse,
    corr,
    desc_0,
    desc_1,
    ghost_sim,
    include_ghost_points=False,
):
    # we cannot include ghost points if we do not have the ghost similarity parameter
    # assert not (include_ghost_points and (ghost_sim is None))
    distance = compute_dist(desc_0, desc_1)
    print(distance.shape)
    torch.Size([70001, 70001])

    exit(0)
    # print(lse.shape, corr.shape, desc_0.shape, desc_1.shape)
    # torch.Size([43780]) torch.Size([1, 43780]) torch.Size([1, 43780, 128]) torch.Size([1, 43780, 128])
    corr = corr.squeeze(0)    
    
    # get mask of valid correspondences
    query_corr = corr >= 0
    # ghost_corr = ~query_corr
    n_corr = query_corr.sum()
    # n_ghost = query_corr.shape[0] - n_corr

    # # make -1 correspondences out-of-bound (for the next get fille)
    # print(desc_0.shape, desc_1.shape)
    # print(corr.shape)
    corr_mask = corr.repeat(desc_1.shape[1], 1).transpose(1,0)

    _desc_1 = torch.where(corr_mask>0, desc_1[corr], 0)


    ################ 
    # this is what they meant
    # idx = corr[43760] 
    # print(desc_1[idx] == _desc_1[43760]) #True

    # aligned dot product
    log_num = torch.bmm(desc_0.unsqueeze(1), _desc_1.unsqueeze(2)).reshape(-1,1)

    # compute log of denominator
    log_den = lse.clone()
    # if ghost_sim is not None:
    #     log_den = torch.logaddexp(log_den, ghost_sim)

    log_p_corr = torch.sum(log_num[query_corr==True]) - torch.sum(log_den[query_corr==True])
    # print(log_p_corr)
    # if include_ghost_points:
    #     log_p_ghost = ghost_sim * n_ghost - torch.sum(log_den[ghost_corr==True])
    # else:
    #     log_p_ghost = 0.0

    normalize = True
    if normalize:
        log_p_corr /= n_corr
        # log_p_ghost /= n_ghost
    else:
        log_p_corr /= query_corr.shape[0]
        # log_p_ghost /= query_corr.shape[0]

    log_p = log_p_corr #+ log_p_ghost
    # print(-log_p)
    return -log_p


def sym_corr_cross_entropy(
    desc_0,
    desc_1,
    corr_0,
    corr_1,
    ghost_sim,
):
    corr_0 = corr_0.squeeze(0)
    corr_1 = corr_1.squeeze(0)

    query_corr_0 = corr_0 >= 0
    query_corr_1 = corr_1 >= 0

    corr_0 = corr_0[query_corr_0] # valid correspondences
    corr_1 = corr_1[query_corr_1]
    # print(corr_0) 
    # print(corr_1)
    # tensor([3275, 3276, 3277, 3807, 3889, 3890, 3891], device='cuda:0')
    # tensor([3275, 3276, 3277, 3807, 3889, 3890, 3891], device='cuda:0')

    sim_0 = torch.matmul(desc_0, desc_1.T)
    sim_0 = torch.softmax(sim_0, dim=1)    

    # idx = corr_0[0]
    # a = distances_0[corr_0]
    # print(a[0])
    # print(distances_0[idx])
    # tensor([1.5284e-05, 1.5331e-05, 1.5378e-05,  ..., 1.3433e-05, 1.5157e-05,
    #     1.5039e-05], device='cuda:0')
    # tensor([1.5284e-05, 1.5331e-05, 1.5378e-05,  ..., 1.3433e-05, 1.5157e-05,
    #     1.5039e-05], device='cuda:0')
    # torch.Size([7, 70001])

    a = sim_0[corr_0]



    exit(0)



    # print(corr_0.shape, corr_1.shape)
    # torch.Size([1, 7001, 2]) torch.Size([1, 7001, 2])

    # print(desc_0.shape)
    def match_descriptors(
        distances,
        max_distance=torch.inf,
        cross_check=True,
        max_ratio=1.0,
    ):
        indices1 = torch.arange(distances.shape[0], device=distances.device)
        # print(distances.shape) #torch.Size([7001, 7001])
        indices2 = torch.argmin(distances, dim=1)
        # print(indices2.shape) #torch.Size([7001])

        if cross_check:
            matches1 = torch.argmin(distances, dim=0)
            mask = indices1 == matches1[indices2]
            indices1 = indices1[mask]
            indices2 = indices2[mask]

    return loss_0 + loss_1


def corr_matching_binary_cross_entropy(
    best_idx_0,
    best_idx_1,
    best_val_0,
    best_val_1,
    corr_0,
    corr_1,
    logits_0,
    logits_1,
    ghost_sim=None,
):
    if ghost_sim is not None:
        best_idx_0 = torch.where(best_val_0 > ghost_sim, best_idx_0, -1)
        best_idx_1 = torch.where(best_val_1 > ghost_sim, best_idx_1, -1)

    best_idx_0, best_idx_1 = keep_mutual_correspondences_only(best_idx_0.to(torch.int32), best_idx_1.to(torch.int32))
    
    # gt positives mask
    gt_mask_0 = corr_0 >= 0
    gt_mask_1 = corr_1 >= 0

    # pred positives mask
    pr_mask_0 = best_idx_0 >= 0
    pr_mask_1 = best_idx_1 >= 0

    # true positives
    tp_mask_0 = torch.logical_and(gt_mask_0, pr_mask_0)
    tp_mask_1 = torch.logical_and(gt_mask_1, pr_mask_1)

    # correct matches
    correct_mask_0 = corr_0 == best_idx_0
    correct_mask_1 = corr_1 == best_idx_1

    loss_0 = correct_mask_0 * torch.nn.functional.softplus(-logits_0) + (
        ~correct_mask_0
    ) * torch.nn.functional.softplus(+logits_0)
    loss_1 = correct_mask_1 * torch.nn.functional.softplus(-logits_1) + (
        ~correct_mask_1
    ) * torch.nn.functional.softplus(+logits_1)

    train_precision = False
    train_recall = True

    assert train_precision or train_recall

    m0 = tp_mask_0.clone()
    m1 = tp_mask_1.clone()

    if train_recall:
        m0 = torch.logical_or(m0, gt_mask_0)
        m1 = torch.logical_or(m1, gt_mask_1)

    if train_precision:
        m0 = torch.logical_or(m0, pr_mask_0)
        m1 = torch.logical_or(m1, pr_mask_1)

    n0 = m0.sum()
    n1 = m1.sum()
    # print(m0.shape, n0.shape, m1.shape, n1.shape)
    # print(m0, n0, m1, n1)
    # torch.Size([1, 43780]) torch.Size([]) torch.Size([1, 43780]) torch.Size([])
    # tensor([[False, False, False,  ..., False, False, False]], device='cuda:1') tensor(344, device='cuda:1') tensor([[False, False, False,  ..., False, False, False]], device='cuda:1') tensor(344, device='cuda:1')

    loss_0 = torch.sum(loss_0[m0==True])
    loss_1 = torch.sum(loss_1[m1==True])

    loss = (loss_0 + loss_1) / (n0 + n1)

    precision = tp_mask_0.sum() / pr_mask_0.sum()
    recall = tp_mask_0.sum() / gt_mask_0.sum()

    return loss, precision, recall


def total_loss(
    desc_0,
    desc_1,
    corr_0,
    corr_1,
    logits_0,
    logits_1,
    ghost_sim,
    block_size,
):
    # info nce loss
    # L_desc
    loss_0 = sym_corr_cross_entropy(
        desc_0.to("cuda:0"),
        desc_1.to("cuda:0"),
        corr_0,
        corr_1,
        ghost_sim,
    )


    # matching loss
    # L_key
    # loss_1, precision, recall = corr_matching_binary_cross_entropy(
    #     argmax_0.unsqueeze(0),
    #     argmax_1.unsqueeze(0),
    #     max_0.unsqueeze(0),
    #     max_1.unsqueeze(0),
    #     corr_0,
    #     corr_1,
    #     logits_0,
    #     logits_1,
    #     ghost_sim,
    # )




    return loss_0.to("cuda:1"), 0


    # return loss_0.to("cuda:1"), loss_1.to("cuda:1"), precision.to("cuda:1"), recall.to("cuda:1")
    # return loss_0.to("cuda:1"), loss_1, precision, recall
