# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

from typing import Union, Dict, List, Tuple, Any, Callable
import logging
import os
import re
import time

import torch

from .hdfs_io import hexists, hmkdir, hcopy
from .torch_io import save as hdfs_torch_save
logger = logging.getLogger(__name__)


class Checkpointer:
    """
    这个类主要是将training checkpointer和state存储到hdfs上.
    """

    def __init__(self,
                 serialization_dir: str = ".output",
                 keep_serialized_model_every_num_seconds: int = None,
                 num_serialized_models_to_keep: int = 20) -> None:
        self._serialization_dir = serialization_dir
        self._keep_serialized_model_every_num_seconds = keep_serialized_model_every_num_seconds
        self._num_serialized_models_to_keep = num_serialized_models_to_keep
        if not hexists(self._serialization_dir):
            hmkdir(self._serialization_dir)

        self._last_permanent_saved_checkpoint_time = time.time()
        self._serialized_paths: List[Tuple[str, str]] = []

    def save_checkpoint(self,
                        epoch: Union[int, str],
                        model_state: Dict[str, Any],
                        training_states: Dict[str, Any],
                        is_best_so_far: bool = False) -> None:
        """
        保存 checkpoint到本地local和remote hdfs中：
        args:
            epoch: 当前训练的epoch数
            model_state: 当前训练model的参数
            training_states: 当前训练的参数
            is_best_so_far: 当前是否save的checkpoint是否为最优
        """
        if self._serialization_dir is not None:
            model_path = os.path.join(
                self._serialization_dir, "model_state_epoch_{}.th".format(epoch))
            training_path = os.path.join(self._serialization_dir,
                                         "training_state_latest.th")
            hdfs_torch_save(model_state, model_path)
            hdfs_torch_save({**training_states, "epoch": epoch}, training_path)

            if is_best_so_far:
                logger.info("Best validation performance so far. "
                            "Copying weights to '%s/best.th'.", self._serialization_dir)
                hcopy(model_path, os.path.join(
                    self._serialization_dir, "best.th"))

            if self._num_serialized_models_to_keep and self._num_serialized_models_to_keep >= 0:
                self._serialized_paths.append((model_path, training_path))
                if len(self._serialized_paths) > self._num_serialized_models_to_keep:
                    paths_to_remove = self._serialized_paths.pop(0)
                    # Check to see if we should keep this checkpoint, if it has been longer
                    # then self._keep_serialized_model_every_num_seconds since the last
                    # kept checkpoint.
                    remove_path = True
                    if self._keep_serialized_model_every_num_seconds is not None:
                        save_time = paths_to_remove[0]
                        time_since_checkpoint_kept = save_time - \
                            self._last_permanent_saved_checkpoint_time
                        if time_since_checkpoint_kept > self._keep_serialized_model_every_num_seconds:
                            # We want to keep this checkpoint.
                            remove_path = False
                            self._last_permanent_saved_checkpoint_time = save_time

    def find_latest_checkpoint(self) -> Tuple[str, str]:
        """
        Return the location of the latest model and training state files.
        If there isn't a valid checkpoint then return None.
        """
        have_checkpoint = (self._serialization_dir is not None and
                           any("model_state_epoch_" in x for x in os.listdir(self._serialization_dir)))

        if not have_checkpoint:
            return None

        serialization_files = os.listdir(self._serialization_dir)
        model_checkpoints = [
            x for x in serialization_files if "model_state_epoch" in x]
        # Get the last checkpoint file.  Epochs are specified as either an
        # int (for end of epoch files) or with epoch and timestamp for
        # within epoch checkpoints, e.g. 5.2018-02-02-15-33-42
        found_epochs = [
            re.search(r"model_state_epoch_([0-9\.\-]+)\.th", x).group(1)
            for x in model_checkpoints
        ]
        int_epochs: Any = []
        print(found_epochs)
        for pieces in found_epochs:
            int_epochs.append([int(float(pieces)), '0'])

        last_epoch = sorted(int_epochs, reverse=True)[0]
        if last_epoch[1] == '0':
            epoch_to_load = str(last_epoch[0])
        else:
            epoch_to_load = '{0}.{1}'.format(last_epoch[0], last_epoch[1])

        model_path = os.path.join(self._serialization_dir,
                                  "model_state_epoch_{}.th".format(epoch_to_load))
        training_state_path = os.path.join(self._serialization_dir,
                                           "training_state_epoch_{}.th".format(epoch_to_load))
        return (model_path, training_state_path)

    def restore_checkpoint(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes a training state (typically consisting of an epoch count and optimizer state),
        which is serialized separately from  model parameters. This function should only be used to
        continue training - if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return empty dicts.

        Returns
        -------
        states: Tuple[Dict[str, Any], Dict[str, Any]]
            The model state and the training state.
        """
        # latest_checkpoint = self.find_latest_checkpoint()

        # if latest_checkpoint is None:
        #     # No checkpoint to restore, start at 0
        #     return {}, {}

        # model_path, training_state_path = latest_checkpoint

        # # Load the parameters onto CPU, then transfer to GPU.
        # # This avoids potential OOM on GPU for large models that
        # # load parameters onto GPU then make a new GPU copy into the parameter
        # # buffer. The GPU transfer happens implicitly in load_state_dict.
        # model_state = torch.load(model_path, map_location=device_mapping(-1))
        # training_state = torch.load(
        #     training_state_path, map_location=device_mapping(-1))
        # return model_state, training_state

    def best_model_state(self) -> Dict[str, Any]:
        """
        load最优的model参数
        """
        if self._serialization_dir:
            logger.info("loading best weights")
            best_model_state_path = os.path.join(
                self._serialization_dir, 'best.th')
            return torch.load(best_model_state_path)
        else:
            logger.info("cannot load best weights without `serialization_dir`, "
                        "so you're just getting the last weights")
            return {}
