# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.

import argparse
import os
import sys
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_image_sampling import DaVinci
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from torch.optim import Optimizer
from torch import optim
from torch.distributed.elastic.multiprocessing.errors import record

from util.checkpointer import Checkpointer
from util.hdfs_io import hmkdir, hcopy
from accelerators.apex_ddp_accelerator import ApexDDPAccelerator

# for dall_e import module, we need to add root path
root_dir = Path(__file__).parent.absolute()
model_dir = root_dir / 'models'
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(model_dir))

MAX_TOKENS = 25

@torch.no_grad()
def train(model, pair_data_loader, optimizer, epoch_info, device, scheduler, config, accelerator, checkpointer, tokenizer):
    # eval - image generation
    model.eval()
    start_epoch, _ = epoch_info
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=50, fmt='{value:.8f}'))

    metric_logger.add_meter('loss', utils.SmoothedValue(
        window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(start_epoch)
    print_freq = 50

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    step_per_epoch = math.ceil(
        config['train_dataset_size'] / (config['batch_size']*world_size))
    current_step = start_epoch * step_per_epoch
    global_step = current_step + 1

    for i, (image, visual_token_image, org_texts, fname) in enumerate(metric_logger.log_every(pair_data_loader, print_freq, header, step_per_epoch, epoch_info)):
        current_epoch = int(global_step/step_per_epoch)

        image = image.to(device, non_blocking=True)
        visual_token_image = visual_token_image.to(device,non_blocking=True)

        prefix_image = None
        prefix_image_small = None
        suffix_image_small = visual_token_image

        org_texts = org_texts * config["num_images"]
        text_full = tokenizer(org_texts, padding='max_length', truncation=True, max_length=MAX_TOKENS, return_tensors="pt").to(device)
        loss, logits = model(image, context=None, gen_text=None, text_full=text_full, prefix_image=prefix_image, suffix_image=suffix_image_small,
                            prefix_image_small=prefix_image_small, visual_token_image=None, use_dalle=True, train=True, decode=False, raw_caption=org_texts, captionindex=i)

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        global_step += 1
        train_stats = {k: "{:.3f}".format(
            meter.global_avg) for k, meter in metric_logger.meters.items()}
        if global_step % step_per_epoch == 0 or global_step % config['checkpoint_frequent'] == 0:
            if utils.is_main_process():
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'epoch': current_epoch,
                             }

                with open("./log.txt", "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@record
def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    if utils.is_main_process():
        print(f"### val_file: {config['val_file']}")
        sys.stdout.flush()

        yaml.dump(config, open('./config.yaml', 'w'))
        hcopy('./config.yaml', args.output_dir)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']

    #### Dataset ####
    print("Creating dataset")
    pair_dataset = [create_dataset('dalle_gen', config)]

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            pair_dataset, [False], num_tasks, global_rank)
    else:
        samplers = [None]
    pair_data_loader = create_loader(pair_dataset, samplers, batch_size=[
                                     config['batch_size']], num_workers=[4], is_trains=[False], collate_fns=[None])[0]

    tokenizer = BertTokenizer.from_pretrained(
        args.encoder, bos_token='[CLS]', eos_token='[SEP]', add_single_sep=False)

    #### Model ####
    print("Creating model")
    model = DaVinci(config=config, encoder=args.encoder, text_decoder=args.text_decoder,
                   tokenizer=tokenizer, init_deit=True, init_dalle=True, device=device)
    model = model.to(device)
    print("DAVINCI have {} paramerters in total".format(
        sum(x.numel() for x in model.parameters())))

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    update_steps_per_epoch = math.ceil(config['train_dataset_size'] / (
        config['batch_size']*world_size) / int(config['accelerator']['GRAD_ACCUMULATE_STEPS']))
    arg_sche['num_warmup_steps'] = arg_sche['warmup_epochs'] * \
        update_steps_per_epoch
    arg_sche['num_training_steps'] = arg_sche['epochs'] * \
        update_steps_per_epoch
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    arg_acc = utils.AttrDict(config['accelerator'])
    accelerator = ApexDDPAccelerator(arg_acc, logger=None)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        start_epoch = checkpoint['epoch']+1
        model.load_state_dict(state_dict, strict=False) # for clip model
        print('load checkpoint from %s' % args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    checkpointer = Checkpointer(args.output_dir)

    print("Start training")
    start_time = time.time()
    epoch_info = (start_epoch, max_epoch)

    train(model, pair_data_loader, optimizer, epoch_info, device, lr_scheduler, config,
          accelerator, checkpointer, tokenizer)
    dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if utils.is_main_process():
        hcopy('./log.txt', args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain_davinci.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--override_cfg', default="",
                        type=str, help="Use ; to separate keys")
    args = parser.parse_args()

    # currently support the override of params at max depth 2
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if args.override_cfg != "":
        override_cfg_str = args.override_cfg.replace(
            ";", "\n").replace(":", ": ")
        override_cfg = yaml.load(override_cfg_str, Loader=yaml.Loader)
        for k, v in override_cfg.items():
            if type(v) == dict:
                for kk, vv in v.items():
                    config[k][kk] = vv
            else:
                config[k] = v
    if not args.output_dir.startswith('hdfs'):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    hmkdir(args.output_dir)
    print("args.output_dir: ", args.output_dir)

    main(args, config)
