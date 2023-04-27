# Write and Paint: Generative Vision-Language Models are Unified Modal Learners (https://arxiv.org/abs/2206.07699)
# Github: https://github.com/shizhediao/DaVinci
# Copyright (c) 2023, ByteDance Inc.
# All rights reserved.


#!/usr/bin/env python
import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from optim.lars import LARS
import math

# from models.model_linearprobe import DaVinciLinearProbe
from models.model_linearprobe import DaVinciLinearProbe
import ruamel.yaml as yaml
from models.tokenization_bert import BertTokenizer
from models.resnet import interpolate_pos_embed

import sys
from pathlib import Path
import utils
from optim import create_optimizer
from scheduler import create_scheduler
from PIL import Image

DATASETS = {
    "celeba": datasets.CelebA,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "emnist": datasets.EMNIST,
    "fakedata": datasets.FakeData,
    "fashionmnist": datasets.FashionMNIST,
    "flickr8k": datasets.Flickr8k,
    "flickr30k": datasets.Flickr30k,
    "inaturalist": datasets.INaturalist,
    "kmnist": datasets.KMNIST,
    "lfwpeople": datasets.LFWPeople,
    "lsun": datasets.LSUN,
    "mnist": datasets.MNIST,
    "omniglot": datasets.Omniglot,
    "places365": datasets.Places365,
    "qmnist": datasets.QMNIST,
    "semeion": datasets.SEMEION,
    "sbu": datasets.SBU,
    "stl10": datasets.STL10,
    "svhn": datasets.SVHN,
    "usps": datasets.USPS,
    #----below are only supported by torch1.11 + torchvision0.12
    "sun397": datasets.SUN397,
    "country211": datasets.Country211,
    "dtd": datasets.DTD,
    "caltech101": datasets.Caltech101,
    "caltech256": datasets.Caltech256,
    "stanfordcars": datasets.StanfordCars,
    "renderedsst2": datasets.RenderedSST2,
    "pcam": datasets.PCAM,
    "oxfordiiitpet": datasets.OxfordIIITPet,
    "flowers102": datasets.Flowers102,
    "food101": datasets.Food101,
    "gtsrb": datasets.GTSRB,
    "fer2013": datasets.FER2013,
    "fgvcaircraft": datasets.FGVCAircraft,
    "eurosat": datasets.EuroSAT,
    "kitti": datasets.Kitti,
}

dataset2nlabels = {
    'imagenet': 1000,
    'food101': 101,
    'cifar10': 10,
    'cifar100': 100,
    'stanfordcars': 196,
    'fgvcaircraft': 102,
    'dtd': 47,
    'oxfordiiitpet': 37,
    'flowers102': 103, # flowers 1 - 102
    'mnist': 10,
    'stl10': 10,
    # 'gtsrb': 43, # data unavailable
    # 'kitti': unclear structure
    'country211': 211,
}

root_dir = Path(__file__).parent.absolute()
model_dir = root_dir / 'models'
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(model_dir))

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
# parser.add_argument('--epochs', default=100, type=int, metavar='N',
#                     help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='./pretrain_coco_vg_6490601_20220429-004728/model_state_epoch_38.th', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--encoder', default='bert-base-uncased')
parser.add_argument('--text_decoder', default='bert-base-uncased')
parser.add_argument('--config', default='./configs/linear_probe.yaml') 
parser.add_argument('--override_cfg', default="", type=str, help="Use ; to separate keys")

best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    # currently support the override of params at max depth 2
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    if args.override_cfg != "":
        override_cfg_str = args.override_cfg.replace(";", "\n").replace(":", ": ")
        override_cfg = yaml.load(override_cfg_str, Loader=yaml.Loader)
        for k, v in override_cfg.items():
            if type(v) == dict:
                for kk, vv in v.items():
                    config[k][kk] = vv
            else:
                config[k] = v

    args.epochs = config['schedular']['epochs']
    args.batch_size = config['batch_size_train']
    print(f"actual lr: {config['optimizer']['lr']}")

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, config))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, config)


def main_worker(gpu, ngpus_per_node, args, config):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create model
    # print("=> creating model '{}'".format(args.arch))
    # model = models.__dict__[args.arch]()
    tokenizer = BertTokenizer.from_pretrained(args.encoder, bos_token='[CLS]', eos_token='[SEP]', add_single_sep=False)
    N_LABELS = dataset2nlabels[config['dataset']]
    model = DaVinciLinearProbe(config=config, encoder=args.encoder, text_decoder=args.text_decoder, tokenizer=tokenizer, n_labels=N_LABELS)
    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location='cpu') 
            state_dict = checkpoint['model']
            # state_dict = checkpoint['state_dict']
            
            for key in list(state_dict.keys())[:]:
                new_key = 'davinci.'+key
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

            # reshape positional embedding to accomodate for image resolution change
            pos_embed_reshaped = interpolate_pos_embed(state_dict['davinci.visual_encoder.pos_embed'],model.davinci.visual_encoder)         
            state_dict['davinci.visual_encoder.pos_embed'] = pos_embed_reshaped     
                    
            msg = model.load_state_dict(state_dict,strict=False)  
            print('loaded checkpoint from %s'%args.pretrained)
            print(msg)  
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    model.fc = torch.nn.Sequential(torch.nn.BatchNorm1d(model.fc.in_features, affine=False, eps=1e-6), model.fc)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    print("config['optimizer']:", config['optimizer'])
    print("args.world_size: ", args.world_size)
    print("args: ", args)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    update_steps_per_epoch = math.ceil(config['train_dataset_size'] / (config['batch_size_train']*args.world_size) / int(config['accelerator']['GRAD_ACCUMULATE_STEPS']))
    arg_sche['num_warmup_steps'] = arg_sche['warmup_epochs'] * update_steps_per_epoch
    arg_sche['num_training_steps'] = arg_sche['epochs'] * update_steps_per_epoch
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    data_path = config['root_dir'] + config['dataset']
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')

    if config['dataset'] == 'imagenet':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    else:
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
                       
    if config['dataset'] == 'mnist':
        train_transform = transforms.Compose([
                transforms.Resize(config['image_res'], interpolation=3),
                transforms.Grayscale(3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transform = transforms.Compose([
            transforms.Resize(config['image_res'], interpolation=3),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])


    if config['dataset'] == 'imagenet':
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        val_dataset = datasets.ImageFolder(valdir, test_transform)
    else:
        if config['dataset'] in ['mnist', 'cifar10', 'cifar100', 'kitti']:
            train_dataset = DATASETS[config['dataset']](
                traindir, train=True, download=True, transform=train_transform)
            val_dataset = DATASETS[config['dataset']](
                valdir, train=False, download=True, transform=test_transform)
        elif config['dataset'] in ['dtd', 'fgvcaircraft', 'food101', 'stanfordcars']:
            train_dataset = DATASETS[config['dataset']](
                traindir, split='train', download=True, transform=train_transform)
            val_dataset = DATASETS[config['dataset']](
                valdir, split='test', download=True, transform=test_transform)
        elif config['dataset'] in ['oxfordiiitpet']:
            train_dataset = DATASETS[config['dataset']](
                traindir, split='trainval', download=True, transform=train_transform)
            val_dataset = DATASETS[config['dataset']](
                valdir, split='test', download=True, transform=test_transform) 
        else:
            train_dataset = DATASETS[config['dataset']](
                traindir, split='train', download=True, transform=train_transform)
            val_dataset = DATASETS[config['dataset']](
                valdir, split='test', download=True, transform=test_transform)     
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args, tokenizer)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args, tokenizer)
        # scheduler.step()

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, tokenizer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            }, is_best)
        print("best_acc1 = ", best_acc1)

def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, tokenizer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.8f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    lr_log = AverageMeter('lr', ':.8f')
    progress = ProgressMeter(
        len(train_loader),
        [lr_log, batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader): # images: [4096, 3, 224, 224], target: [4096]
        # measure data loading time
        data_time.update(time.time() - end)

        # FROM MAE: we use a per iteration (instead of per epoch) lr scheduler
        # adjust_learning_rate(optimizer, i / len(train_loader) + epoch, args)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        # compute output
        output = model(images, train=True) # output: [4096, 1000]
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        lr_log.update(optimizer.param_groups[0]["lr"], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        scheduler.step()   
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, tokenizer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
                
            # compute output
            output = model(images, train=False)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
