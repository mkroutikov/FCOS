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


class FCOSRectangleLoss:
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
        l, t, r, b = logits['l'], logits['t'], logits['r'], logits['b']
        assert len(l) == 5  # 5 layers in image pyramid
        assert len(t) == 5
        assert len(r) == 5
        assert len(b) == 5

        batch_size = l[0].shape[0]
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

def compute_targets_for_locations(locations, targets, stripe_width=10):
    focus = []
    regression = []
    xs, ys = locations[:, 0], locations[:, 1]

    import pdb; pdb.set_trace()
    for target in targets:
        assert target.mode == "xyxy"

        x = compute_rectangle_stripe_targets(locations, targets.bbox, stripe_width=stripe_width)

        focus.append(x['focus'])
        regression.append(x['regression'])

    return focus, regression


def compute_rectangle_stripe_targets(locations, box, stripe_width=10):
    import pdb; pdb.set_trace()
    xs, ys = locations[:, 0], locations[:, 1]

    l = xs - box[0]
    t = ys - box[1]
    r = box[2] - xs
    b = box[3] - ys

    # build left stripe
    mask = ((torch.abs(l) <= stripe_width) * (t >= 0) * (b >= 0)).float()
    lc = (stripe_width - torch.abs(l)) * mask  # presence mask
    l = l * mask  # regression mask

    # build top stripe
    mask = ((torch.abs(t) <= stripe_width) * (l >= 0) * (r >= 0)).float()
    tc = (stripe_width - torch.abs(t)) * mask  # presence mask
    t = t * mask

    # build right stripe
    mask = ((torch.abs(r) <= stripe_width) * (t >= 0) * (b >= 0)).float()
    rc = (stripe_width - torch.abs(r)) * mask  # presence mask
    r = r * mask

    # build bottom stripe
    mask = ((torch.abs(b) <= stripe_width) * (l >= 0) * (r >= 0)).float()
    bc = (stripe_width - torch.abs(b)) * mask  # presence mask
    b = b * mask

    foc = torch.stack([lc, tc, rc, bc], dim=1)
    reg = torch.stack([l, t, r, b], dim=1)

    return {
        'focus': foc,
        'regression': reg,
    }



def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                    (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)
