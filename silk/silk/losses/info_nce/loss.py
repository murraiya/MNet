# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import math
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

    return -log_p


def sym_corr_cross_entropy(
    lse_0,
    lse_1,
    desc_0,
    desc_1,
    corr_0,
    corr_1,
    ghost_sim,
):
    # print(corr_0.shape, corr_1.shape)
    # torch.Size([1, 7001, 2]) torch.Size([1, 7001, 2])

    loss_0 = asym_corr_cross_entropy(
        lse_0,
        corr_0,
        desc_0.clone(),
        desc_1.clone(),
        ghost_sim=ghost_sim,
    )
    loss_1 = asym_corr_cross_entropy(
        lse_1,
        corr_1,
        desc_1,
        desc_0,
        ghost_sim=ghost_sim,
    )

    # if math.isnan(loss_0) or math.isnan(loss_1):
    #     print(loss_0, loss_1)
    #     print("nan detected, shutdown")
    #     exit(0)

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
    # print(desc_0.shape, desc_1.shape)
    # torch.Size([7001, 128]) torch.Size([7001, 128])

    x0x1 = desc_0 @ desc_1.T

    lse_0 = torch.logsumexp(x0x1, axis=1) #soft version of max
    lse_1 = torch.logsumexp(x0x1, axis=0)
        
    # info nce loss
    # L_desc
    loss_0 = sym_corr_cross_entropy(
        lse_0,
        lse_1,
        desc_0.to("cuda:0"),
        desc_1.to("cuda:0"),
        corr_0,
        corr_1,
        ghost_sim,
    )

    return loss_0.to("cuda:1")


    # return loss_0.to("cuda:1"), loss_1.to("cuda:1"), precision.to("cuda:1"), recall.to("cuda:1")
    # return loss_0.to("cuda:1"), loss_1, precision, recall
