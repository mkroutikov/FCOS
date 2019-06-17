from collections import OrderedDict
from maskrcnn_benchmark.modeling.backbone import resnet, fpn as fpn_module
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from torch import nn
import torch


def build_resnet_fpn_p3p7_backbone(stem_out_channels=64, in_channels_stage2=256, out_channels=256, use_c5=False, use_gn=False, use_relu=False):
    stem_module = resnet.StemWithFixedBatchNorm(stem_out_channels)
    stage_specs = [  # R-50-FPN-RETINANET
        resnet.StageSpec(index=1, block_count=3, return_features=True),
        resnet.StageSpec(index=2, block_count=4, return_features=True),
        resnet.StageSpec(index=3, block_count=6, return_features=True),
        resnet.StageSpec(index=4, block_count=3, return_features=True)
    ]
    transformation_module = resnet.BottleneckWithFixedBatchNorm

    body = resnet.ResNetLight(stem_module, stage_specs, transformation_module)

    in_channels_p6p7 = in_channels_stage2 * 8 if use_c5 else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(use_gn, use_relu),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

class FCOSModuleLight(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, in_channels,
        num_classes=80,
        num_convs=4,
        prior_prob=0.01,
        inference_th=0.05,
        pre_nms_top_n=1000,
        nms_th=0.6,
        fpn_post_nms_top_n=100,
        loss_gamma=2.0,
        loss_alpha=0.25,
        fpn_strides=[8, 16, 32, 64, 128],
    ):
        super(FCOSModuleLight, self).__init__()

        self.head = FCOSHead(
            num_classes=num_classes,
            num_convs=num_convs,
            prior_prob=prior_prob,
            in_channels=in_channels
        )

        self.box_selector_test = FCOSPostProcessor(
            pre_nms_thresh=inference_th,
            pre_nms_top_n=pre_nms_top_n,
            nms_thresh=nms_th,
            fpn_post_nms_top_n=fpn_post_nms_top_n,
            min_size=0,
            num_classes=num_classes+1
        )

        self.loss_evaluator = FCOSLossComputation(loss_gamma, loss_alpha)

        self.fpn_strides = fpn_strides

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(
                locations, box_cls, box_regression,
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


class FCOSModel(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, num_classes=80):
        super(FCOSModel, self).__init__()

        self.backbone = build_resnet_fpn_p3p7_backbone()
        self.rpn = FCOSModuleLight(in_channels=self.backbone.out_channels, num_classes=num_classes)
        self.roi_heads = []

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.training:
            return proposal_losses
        else:
            return proposals

