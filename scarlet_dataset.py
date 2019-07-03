# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simple dataset class that wraps a list of path names
"""

from PIL import Image

from maskrcnn_benchmark.structures.bounding_box import BoxList
from ilabs.curate import ic
import torch
import lxml.etree as et
import transforms as T


INPUT_PIXEL_MEAN = (102.9801, 115.9465, 122.7717)
INPUT_PIXEL_STD  = (1., 1., 1.)

RESNET50_MEAN = [0.485, 0.456, 0.406]
RESNET50_STD  = [0.229, 0.224, 0.225]

class Scarlet300Dataset:
    def __init__(self, split):
        assert split in ('test', 'train')
        self.split = split

        dataset = ic.get_dataset('ilabs.vision', 'scarlet300')
        self._images = sorted(x for x in dataset[split] if x.endswith('.png'))
        self._boxes = [build_boxlist(x[:-4] + '.xml') for x in self._images]

        if split == 'test':
            self._transform = T.Compose([
                T.PadToDivisibility(32),
                T.RandomCrop(32, 32),
                T.ToTensor(),
                T.Normalize(mean=RESNET50_MEAN, std=RESNET50_STD, to_bgr255=False),
                T.validate_target,
                #T.MakeMaskChannel(),
            ])
        else:
            self._transform = T.Compose([
                T.PadToDivisibility(32),
                T.CenterCrop(32, 32),
                T.ToTensor(),
                T.Normalize(mean=RESNET50_MEAN, std=RESNET50_STD, to_bgr255=False),
                #T.MakeMaskChannel(),
            ])

    def __getitem__(self, item):
        img = Image.open(self._images[item]).convert("RGB")
        target = self._boxes[item]

        img, target = self._transform(img, target)

        return img, target

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
    assert (boxes[:,1] == boxes[:,3]).nonzero().sum() == 0

    classes = torch.tensor([1]*len(boxes), dtype=torch.int32)

    return {
        'boxes': boxes,
        'labels': classes,
    }



if __name__ == '__main__':
    from transforms import PadToDivisibility, RandomCrop
    from matplotlib import pyplot as plt
    from PIL import Image, ImageDraw

    transform = RandomCrop(32, 32)
    dataset = Scarlet300Dataset(split='train', transforms=transform)

    for i in range(5):
        image, target = dataset[i]

        draw = ImageDraw.Draw(image)
        for xy in target.convert('xyxy').bbox:
            x0, y0, x1, y1 = xy[0], xy[1], xy[2], xy[3]
            draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0))

        plt.figure()
        plt.imshow(image)

    plt.show()