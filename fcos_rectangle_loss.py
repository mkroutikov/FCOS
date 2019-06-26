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

    def __init__(self, gamma, alpha, stripe_width=32):
        self.cls_loss_func = SigmoidFocalLoss(gamma, alpha)
        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.stripe_width = stripe_width

    def __call__(self, logits, targets):
        locations = logits['locations']
        focus = logits['focus']
        regression = logits['regression']

        batch_size = len(targets)

        focus_losses = []
        regression_losses = []
        for i in range(batch_size):
            target = targets[i]
            for j in range(5):  # Pyramid
                loc = locations[j]
                predicted_focus = focus[j][i].view(4, -1).permute(1, 0)
                predicted_regression = regression[j][i].view(4, -1).permute(1, 0)
                if target.bbox.shape[0] > 0:
                    assert target.bbox.shape[0] == 1
                    result = compute_rectangle_stripe_targets(loc, target.bbox[0], stripe_width=self.stripe_width)
                    focus_loss = torch.abs(result['focus'] - predicted_focus).mean().unsqueeze(0)
                    regression_loss = torch.abs(result['regression'] - predicted_regression).mean().unsqueeze(0)
                else:
                    focus_loss = torch.abs(predicted_focus).mean().unsqueeze(0)
                    regression_loss = torch.abs(predicted_regression).mean().unsqueeze(0)

                focus_losses.append(focus_loss)
                regression_losses.append(regression_loss)

        return {
            'loss_focus': torch.cat(focus_losses).mean(),
            'regression_loss': torch.cat(regression_losses).mean(),
        }

def compute_rectangle_stripe_targets(locations, box, stripe_width=10):
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
