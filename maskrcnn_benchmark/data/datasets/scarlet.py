# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simple dataset class that wraps a list of path names
"""

from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList
from ilabs.curate import ic
import torch
import lxml.etree as et


class Scarlet300Dataset:
    def __init__(self, split, transforms=None):
        assert split in ('test', 'train')
        self.split = split
        self.transforms = transforms

        dataset = ic.get_dataset('ilabs.vision', 'scarlet300')
        self._images = sorted(x for x in dataset[split] if x.endswith('.png'))
        self._boxes = [build_boxlist(x[:-4] + '.xml') for x in self._images]

    def __getitem__(self, item):
        img = Image.open(self._images[item]).convert("RGB")
        target = self._boxes[item]

        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, item

    def __len__(self):
        return len(self._images)

    def get_img_info(self, item):
        width, height = self._boxes[item].size
        return dict(width=width, height=height)


def build_boxlist(fname):

    with open(fname, 'rb') as f:
        xml = et.fromstring(f.read())

    # image dimensions
    width, height = (int(xml.attrib[x]) for x in ['width', 'height'])

    def xyxy(elt):
        return [float(elt.attrib[x]) for x in 'ltrb']

    boxes = [xyxy(elt) for elt in xml.findall('.//block')]
    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
    target = BoxList(boxes, (width, height), mode="xyxy")

    classes = torch.tensor([1]*len(boxes), dtype=torch.int32)
    target.add_field('labels', classes)

    return target
