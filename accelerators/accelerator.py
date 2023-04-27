# -*- coding: utf-8 -*-
'''
Created on Feb-19-21 16:36
accelerator.py
Description: accelerators的基类，便于后续其他加速方案的接入。
'''

from logging import Logger

import torch
from torch.optim import Optimizer

Net = torch.nn.Module


class Accelerator:
    """
    Accelerator是所有accelerators的基类，新添加的accelerator需要继承该类。
    """

    def __init__(self, cfg, logger) -> None:
        self.cfg = cfg
        self.logger = logger
    
    def set_up(self, model: Net):
        raise NotImplementedError("Set Up method not implement in Accelerator, please check! ")

    def broadcast(self):
        raise NotImplementedError("Broadcast method not implement in Accelerator, please check! ")

    def backward_step(self, loss: torch.Tensor):
        loss.backward()

    def optimizer_step(self, optimizer: Optimizer, model: Net, grad_norm: float) -> float:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                        grad_norm)
        return float(total_norm)
    
