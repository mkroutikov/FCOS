# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simple dataset class that wraps a list of path names
"""

from PIL import Image, ImageDraw

from maskrcnn_benchmark.structures.bounding_box import BoxList
from ilabs.curate import ic
import torch
import lxml.etree as et


class Scarlet300MaskDataset:
    def __init__(self, split, transforms=None, single_block=False):
        assert split in ('test', 'train')
        self.split = split
        self.transforms = transforms

        dataset = ic.get_dataset('ilabs.vision', 'scarlet300')
        self._images = sorted(x for x in dataset[split] if x.endswith('.png'))
        self._boxes = [build_boxlist(x[:-4] + '.xml') for x in self._images]

        boxlens = [len(b) for b in self._boxes]

        self._index = []
        for i in range(len(self._images)):
            for j in range(boxlens[i] + 1):
                self._index.append( (i, j) )

    def __getitem__(self, item):
        i, j = self._index[item]

        img = Image.open(self._images[i]).convert("RGB")
        boxes = self._boxes[i].convert('xyxy')
        mask = Image.new('L', img.size)
        draw = ImageDraw.Draw(mask)
        for xy in boxes.bbox[:j]:
            x0, y0, x1, y1 = xy[0], xy[1], xy[2], xy[3]
            draw.rectangle((x0, y0, x1, y1), fill=255)

        if j < len(boxes):
            idx = torch.tensor([j])
            target = boxes[idx].clip_to_image(remove_empty=True)
        else:
            idx = torch.tensor([], dtype=torch.int64)
            target = boxes[idx].clip_to_image(remove_empty=True)

        if self.transforms is not None:
            (img, mask, target), _ = self.transforms(img, mask, target)

        return img, mask, target, item

    def __len__(self):
        return len(self._images)

    def get_img_info(self, item):
        width, height = self._boxes[item].size
        return dict(width=width, height=height)


def build_boxlist(fname, single_block=False):

    with open(fname, 'rb') as f:
        xml = et.fromstring(f.read())

    # image dimensions
    width, height = (int(xml.attrib[x]) for x in ['width', 'height'])

    def xyxy(elt):
        return [float(elt.attrib[x]) for x in 'ltrb']

    boxes = [xyxy(elt) for elt in xml.findall('.//block')]
    if single_block:
        boxes = boxes[:1]
    boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
    target = BoxList(boxes, (width, height), mode="xyxy")

    classes = torch.tensor([1]*len(boxes), dtype=torch.int32)
    target.add_field('labels', classes)

    return target


if __name__ == '__main__':
    from transforms_mask import PadToDivisibility, RandomCrop, Compose, ToTensor, Normalize, MakeMaskChannel
    from matplotlib import pyplot as plt
    from PIL import Image, ImageDraw

    transform = Compose([
        PadToDivisibility(32),
        RandomCrop(32, 32),
        ToTensor(),
        Normalize(mean=(102.9801, 115.9465, 122.7717), std=(1., 1., 1.)),
        MakeMaskChannel(),
    ])
    dataset = Scarlet300MaskDataset(split='train', transforms=transform)

    for i in range(15):
        image, mask, target, _ = dataset[i]
        continue

        draw = ImageDraw.Draw(image)
        for xy in target.convert('xyxy').bbox:
            x0, y0, x1, y1 = xy[0], xy[1], xy[2], xy[3]
            draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0))

        plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.subplot(122)
        plt.imshow(mask)

    plt.show()