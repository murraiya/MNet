# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional

import silk.losses.info_nce.loss as jax_loss
import torch



def total_loss_reduction(
    desc_0,
    desc_1,
    corr_0,
    corr_1,
    logits_0,
    logits_1,
    ghost_sim=None,
    block_size=None,
):
    
    # loss_0, loss_1, precision, recall = jax_loss.total_loss(
    loss_0 = jax_loss.total_loss(
        desc_0.to("cuda:0"),
        desc_1.to("cuda:0"),
        corr_0.to("cuda:0"),
        corr_1.to("cuda:0"),
        logits_0.to("cuda:0"),
        logits_1.to("cuda:0"),
        ghost_sim,
        block_size,
    )

    return loss_0.mean() #, loss_1.mean(), precision, recall



class Loss(torch.nn.Module):
    def __init__(
        self,
        block_size: Optional[int] = None,
        device: str = "cuda:0",
        temperature: float = 0.1,
    ) -> None:
        super().__init__()

        self.cnt = 0
        self._block_size = block_size
        self.device = device
        self._temperature_sqrt_inv = 1.0 / math.sqrt(temperature)

    def __call__(
        self,
        sparse_positions,
        sparse_descriptors,
        desc_0,
        desc_1,
        corr_0,
        corr_1,
        logits_0,
        logits_1,
        ghost_sim=None,
    ):
        self.cnt+=1
        # if self.cnt>1: exit(0)
        # if runs til here, one round from computing loss to updating gradients are fine, and maybe 
        # last loss remained in the memory raising troubles 

        # desc_0 = desc_0 * self._temperature_sqrt_inv
        # desc_1 = desc_1 * self._temperature_sqrt_inv
        # print(desc_0.shape, desc_0.dtype)
        # print(sparse_descriptors[0].shape, sparse_descriptors[0].dtype)
      
        desc_0 = sparse_descriptors[0] * self._temperature_sqrt_inv
        desc_1 = sparse_descriptors[1] * self._temperature_sqrt_inv

        return total_loss_reduction(
            desc_0,
            desc_1,
            corr_0,
            corr_1,
            logits_0,
            logits_1,
            ghost_sim,
            block_size=self._block_size,
        )
        # return total_loss_reduction(
        #     desc_0.clone(),
        #     desc_1.clone(),
        #     corr_0.clone(),
        #     corr_1.clone(),
        #     logits_0.clone(),
        #     logits_1.clone(),
        #     ghost_sim,
        #     block_size=self._block_size,
        # )
