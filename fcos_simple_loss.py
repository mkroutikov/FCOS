"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn

from maskrcnn_benchmark.modeling.rpn.utils import concat_box_prediction_layers
from maskrcnn_benchmark.layers import IOULoss
from maskrcnn_benchmark.layers import SigmoidFocalLoss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


INF = 100000000


class FCOSSimpleLoss:
    """
    This class computes the FCOS losses.
    """

    def __init__(self, gamma, alpha):
        self.cls_loss_func = SigmoidFocalLoss(gamma, alpha)
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()


    def __call__(self, logits, targets):
        locations, box_cls, box_regression, centerness = logits['locations'], logits['box_cls'], logits['box_regression'], logits['centerness']

        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets = prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / (pos_inds.numel() + N)  # add N to avoid dividing by a zero

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        if pos_inds.numel() > 0:
            centerness_targets = compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return {
            'loss_cls': cls_loss,
            'loss_reg': reg_loss,
            'loss_centerness': centerness_loss,
        }


def prepare_targets(points, targets):
    num_points_per_level = [len(points_per_level) for points_per_level in points]
    points_all_level = torch.cat(points, dim=0)
    labels, reg_targets = compute_targets_for_locations(
        points_all_level, targets
    )

    for i in range(len(labels)):
        labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
        reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

    labels_level_first = []
    reg_targets_level_first = []
    for level in range(len(points)):
        labels_level_first.append(
            torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
        )
        reg_targets_level_first.append(
            torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
        )

    return labels_level_first, reg_targets_level_first

def compute_targets_for_locations(locations, targets):
    labels = []
    reg_targets = []
    xs, ys = locations[:, 0], locations[:, 1]

    for im_i in range(len(targets)):
        targets_per_im = targets[im_i]
        assert targets_per_im.mode == "xyxy"
        bboxes = targets_per_im.bbox

        if bboxes.shape[0] == 0:  # no objects!
            labels.append(torch.zeros(locations.shape[0], 1, dtype=torch.int32, device=bboxes.device))
            reg_targets.append(torch.zeros(locations.shape[0], 1, 4, dtype=torch.float32, device=bboxes.device))
            continue

        assert bboxes.shape[0] == 1  # no more than one box!

        labels_per_im = targets_per_im.get_field("labels")
        area = targets_per_im.area()

        # relative box size for each location
        l = xs[:, None] - bboxes[:, 0][None]
        t = ys[:, None] - bboxes[:, 1][None]
        r = bboxes[:, 2][None] - xs[:, None]
        b = bboxes[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        labels_per_im = labels_per_im.repeat(len(locations)).unsqueeze(1)
        mask = (reg_targets_per_im.min(-1)[0] >= 0.).int()
        labels_per_im *= mask
        labels.append(labels_per_im)
        reg_targets.append(reg_targets_per_im)

    return labels, reg_targets

def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                    (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)
