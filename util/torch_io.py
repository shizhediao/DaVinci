# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' torch model hdfs io warpper '''

import io
import torch

from .hdfs_io import hopen


def load(filepath: str, **kwargs):
    """ load model """
    if not filepath.startswith("hdfs://"):
        return torch.load(filepath, **kwargs)
    with hopen(filepath, "rb") as reader:
        accessor = io.BytesIO(reader.read())
        state_dict = torch.load(accessor, **kwargs)
        del accessor
        return state_dict


def save(obj, filepath: str, **kwargs):
    """ save model """
    if filepath.startswith("hdfs://"):
        with hopen(filepath, "wb") as writer:
            torch.save(obj, writer, **kwargs)
    else:
        torch.save(obj, filepath, **kwargs)
