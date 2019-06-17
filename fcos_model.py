import math
from collections import OrderedDict
from maskrcnn_benchmark.modeling.backbone import resnet, fpn as fpn_module
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from maskrcnn_benchmark.layers import Scale

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


class FCOSModel(nn.Module):
    def __init__(self, num_classes=80):
        super(FCOSModel, self).__init__()

        self.backbone = build_resnet_fpn_p3p7_backbone()
        self.rpn = FCOSModuleLight(in_channels=self.backbone.out_channels, num_classes=num_classes)

    def forward(self, images):
        """
        Arguments:
            images (ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns: TODO

        """
        features = self.backbone(images.tensors)
        return self.rpn(images, features)


class FCOSModuleLight(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, in_channels,
        num_classes=80,
        num_convs=4,
        prior_prob=0.01,
        fpn_strides=[8, 16, 32, 64, 128],
    ):
        super(FCOSModuleLight, self).__init__()

        self.head = FCOSHead(
            num_classes=num_classes,
            num_convs=num_convs,
            prior_prob=prior_prob,
            in_channels=in_channels
        )

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

        return {
            'locations'     : locations,
            'box_cls'       : box_cls,
            'box_regression': box_regression,
            'centerness'    : centerness,
        }

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

        return torch.stack((shift_x, shift_y), dim=1) + stride // 2


class FCOSHead(torch.nn.Module):
    def __init__(self, num_classes, num_convs, prior_prob, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.

        cls_tower = []
        bbox_tower = []
        for i in range(num_convs):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(cls_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(self.bbox_tower(feature))
            )))
        return logits, bbox_reg, centerness
