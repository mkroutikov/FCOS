# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import logging
import glob
import json

import torch
from maskrcnn_benchmark.data.collate_batch import BatchCollator
from maskrcnn_benchmark.data import samplers
from maskrcnn_benchmark.solver import WarmupMultiStepLR
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, get_world_size, is_main_process
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.data import transforms as T
import transforms_mask as TTT
from scarlet_mask_dataset import Scarlet300MaskDataset
from fcos_model import FCOSModel
from fcos_simple_loss import FCOSSimpleLoss
from fcos_post_processor import FCOSPostProcessor
from maskrcnn_benchmark.structures.image_list import to_image_list
import torch.distributed as dist
from sane_demo import distill_module

from summary_writer import TensorboardSummary


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def make_optimizer(named_parameters, base_lr=0.001, weight_decay=0.0001, bias_lr_factor=2, weight_decay_bias=0, momentum=0.9):
    params = []
    for key, value in named_parameters:
        if not value.requires_grad:
            continue
        lr = base_lr
        weight_decay = weight_decay
        if "bias" in key:
            lr = base_lr * bias_lr_factor
            weight_decay = weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    return torch.optim.SGD(params, lr, momentum=momentum)


def build_transforms(is_train=True, min_size=800, max_size=1333, flip_prob=0.5, input_pixel_mean=(102.9801, 115.9465, 122.7717), input_pixel_std=(1., 1., 1.), convert_to_bgr255=True):
    if is_train:
        return TTT.Compose(
            [
                TTT.PadToDivisibility(32),
                TTT.RandomCrop(32, 32),
                TTT.ToTensor(),
                TTT.Normalize(mean=input_pixel_mean, std=input_pixel_std, to_bgr255=convert_to_bgr255),
                TTT.MakeMaskChannel(),
            ]
        )
    else:
        return TTT.Compose(
            [
                TTT.PadToDivisibility(32),
                TTT.ToTensor(),
                TTT.Normalize(mean=input_pixel_mean, std=input_pixel_std, to_bgr255=convert_to_bgr255),
                TTT.MakeMaskChannel(),
            ]
        )


def make_train_data_loader(is_distributed=False, start_iter=0, image_size_divisibility=32, num_workers=1, batch_size=2, num_iters=90000, shuffle=True, single_block=False):
    num_gpus = get_world_size()
    assert batch_size % num_gpus == 0, "batch_size ({}) must be divisible by the number "
    "of GPUs ({}) used.".format(batch_size, num_gpus)

    images_per_gpu = batch_size // num_gpus
    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    transforms = build_transforms(is_train=True)

    dataset = Scarlet300MaskDataset('train', transforms=transforms)

    if is_distributed:
        sampler = samplers.DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)

    batch_sampler = samplers.IterationBasedBatchSampler(
        torch.utils.data.sampler.BatchSampler(
            sampler, images_per_gpu, drop_last=False
        ),
        num_iters,
        start_iter
    )

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=TTT.BatchCollator(),
    )

    return data_loader

def train(
    output_dir,
    local_rank,
    distributed=False,
    save_every=2500,
    resume=None,
    warmup_milestones = (60000, 80000),
    warmup_gamma=0.1,
    warmup_factor=0.3333,
    warmup_iters=500,
    warmup_method='constant',
    print_every=20,
    loss_gamma=2.0,
    loss_alpha=0.25,
    image_every=100,
    inference_th=0.05,
    pre_nms_top_n=1000,
    nms_th=0.6,
    fpn_post_nms_top_n=100,
    batch_size=10,
    base_lr=0.001,
    weight_decay=0.0001,
    single_block=False,
    fine_tune=False,
):
    params = locals()

    if is_main_process():
        with open(output_dir + '/params.json', 'w') as f:
            json.dump(params, f, indent=4)

    model = FCOSModel(num_classes=1)
    device = torch.device('cuda:%d' % local_rank if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = FCOSSimpleLoss(loss_gamma, loss_alpha)

    for x in model.backbone.parameters():
        x.requires_grad = False

    optimizer = make_optimizer(model.rpn.named_parameters(), base_lr=base_lr, weight_decay=weight_decay)
    scheduler = WarmupMultiStepLR(
        optimizer,
        warmup_milestones,
        gamma=warmup_gamma,
        warmup_factor=warmup_factor,
        warmup_iters=warmup_iters,
        warmup_method=warmup_method,
    )

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=inference_th,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_th,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=2  # here we count background??? -MK
    )

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    start_iter = 0
    if resume is not None:
        assert os.path.exists(resume)
        state_dict = torch.load(resume, map_location=device)
        model_state_dict = distill_module(state_dict['model'])
        fix_34_channels(model_state_dict)
        model.load_state_dict(model_state_dict)
        if not fine_tune:
            optimizer.load_state_dict(state_dict['optimizer'])
            scheduler.load_state_dict(state_dict['scheduler'])
            start_iter = state_dict['iteration']

    data_loader = make_train_data_loader(
        is_distributed=distributed,
        start_iter=start_iter,
        batch_size=batch_size,
        single_block=single_block,
    )

    if is_main_process():
        summary = TensorboardSummary(logdir=output_dir)

    logging.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)

    model.train()
    for iteration, (images, _, targets, _) in enumerate(data_loader, start_iter):

        scheduler.step()

        images = images.to(device)
        images = to_image_list(images)

        targets = [target.to(device) for target in targets]

        logits = model(images.tensors)
        loss_dict = criterion(logits, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (iteration+1) % print_every == 0 or (iteration+1) == max_iter:
            lr = optimizer.param_groups[0]["lr"]

            for name,value in loss_dict_reduced.items():
                summary.add_scalar(name, value, global_step=iteration+1)

            summary.add_scalar('lr', lr, global_step=iteration+1)

            cuda_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 if torch.cuda.is_available() else 0
            logging.info(
                meters.delimiter.join(
                    [
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    iter=iteration+1,
                    meters=str(meters),
                    lr=lr,
                    memory=cuda_mem,
                )
            )
        if (iteration + 1) % save_every == 0 or iteration + 1 == max_iter:
            if is_main_process() == 0:  # only master process saves model in distributed settings
                fname = os.path.join(output_dir, "model_{:07d}.pth".format(iteration+1))
                model_state_dict = model.state_dict()
                if 'module' in model_state_dict:  # unwrap models wrapped in DistributedDataParallel()
                    model_state_dict = model_state_dict['module']
                torch.save({
                    'model': model_state_dict,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': iteration+1,
                }, fname)

        if (iteration+1) % image_every == 0 and is_main_process():
            with torch.no_grad():
                predictions = box_selector(logits, images.image_sizes)

            summary.visualize_box(images.tensors[0], targets[0], predictions[0].to(torch.device('cpu')), iteration+1)

    return model


def experiment_dir(base_dir='runs'):
    os.makedirs(base_dir, exist_ok=True)

    experiments = glob.glob(base_dir + '/[0-9][0-9][0-9][0-9]')
    if not experiments:
        experiment_no = 1
    else:
        experiment_no = max(int(os.path.basename(x)) for x in experiments) + 1

    dirname = base_dir + '/%04d' % experiment_no
    os.makedirs(dirname)

    logging.info('Experiment directory: %s', dirname)

    return dirname


def fix_34_channels(state_dict):
    weight = state_dict['backbone.body.stem.conv1.weight']
    if weight.shape[1] == 4:
        return  # all good, nothing to do
    assert weight.shape[1] == 3  # RGB - three input channels

    extra_shape = list(weight.shape)
    extra_shape[1] = 1  # adding one more channel
    extra = torch.zeros(*extra_shape, device=weight.device).normal_(0., 0.001)

    new_weight = torch.cat([weight, extra], dim=1)
    state_dict['backbone.body.stem.conv1.weight'] = new_weight


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--output_dir', default='runs', help='where to write models and stats')
    parser.add_argument('--resume', help='filename of a model to resume training with')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--single-block", action='store_true', default=False)
    parser.add_argument("--fine-tune", "-ft", action='store_true', default=False, help="do not restore optimizer and scheduler state")

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    if is_main_process():
        logging.basicConfig(level=logging.INFO)
        output_dir = experiment_dir(base_dir=args.output_dir)
        logger = setup_logger("fcos", output_dir, get_rank())
        logger.info("Using {} GPUs".format(num_gpus))
        logger.info(args)
    else:
        output_dir = None

    train(
        output_dir,
        args.local_rank,
        args.distributed,
        resume=args.resume,
        base_lr=args.lr,
        batch_size=args.batch_size,
        single_block=args.single_block,
        fine_tune=args.fine_tune,
    )


if __name__ == "__main__":
    main()
