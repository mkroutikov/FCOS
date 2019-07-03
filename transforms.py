from PIL import Image, ImageOps
import random
from torchvision.transforms import functional as F
import torch


def pad_boxes(boxes, l, t, r, b):
    x0, y0, x1, y1 = boxes.unbind(1)

    return torch.stack([x0+l, y0+t, x1+l, y1+t], dim=1)


def crop_boxes(boxes, l, t, r, b):
    x0, y0, x1, y1 = boxes.unbind(1)

    x0 = torch.clamp(x0 - l, min=0)
    y0 = torch.clamp(y0 - t, max=r-l)
    x1 = torch.clamp(x1 - l, min=0)
    y1 = torch.clamp(y1 - t, max=b-t)

    return torch.stack([x0, y0, x1, y1], dim=1)


def validate_target(image, target):
    boxes = target['boxes']

    assert (boxes[:,1] == boxes[:,3]).sum() == 0

    return image, target

class PadToDivisibility:
    '''Pads image (and target) to ensure height and width are divisible by |divisibility| '''

    def __init__(self, divisibility=32):
        self.divisibility = divisibility

    def __call__(self, image, target):
        w, h = image.size

        d = self.divisibility
        height = ((h + d - 1) // d) * d
        width = ((w + d - 1) // d) * d

        l = (width - w) // 2
        r = width - w - l
        t = (height - h) // 2
        b = height - h - t

        image = ImageOps.expand(image, (l, t, r, b), fill='white')
        if target is not None:
            target['boxes'] = pad_boxes(target['boxes'], l, t, r, b)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomCrop:
    '''Randomly crops an image to a smaller one by |delta_x|, |delta_y|'''

    def __init__(self, delta_x=0, delta_y=0):
        self.delta_x = delta_x
        self.delta_y = delta_y

    def __call__(self, image, target):
        dx = self.delta_x
        dy = self.delta_y

        if dx == 0 and dy == 0:
            return image, target

        offx = random.randint(0, dx)
        offy = random.randint(0, dy)

        w, h = image.size
        image = image.crop((offx, offy, offx+w-dx, offy+h-dy))
        if target is not None:
            target['boxes'] = crop_boxes(target['boxes'], offx, offy, offx+w-dx, offy+h-dy)

        return image, target


class CenterCrop:
    '''Randomly crops an image to a smaller one by |delta_x|, |delta_y|'''

    def __init__(self, delta_x=0, delta_y=0):
        self.delta_x = delta_x
        self.delta_y = delta_y

    def __call__(self, image, target):
        dx = self.delta_x
        dy = self.delta_y

        if dx == 0 and dy == 0:
            return image, target

        offx = dx // 2
        offy = dy // 2

        w, h = image.size
        image = image.crop((offx, offy, offx+w-dx, offy+h-dy))
        if target is not None:
            target['boxes'] = crop_boxes(target['boxes'], offx, offy, offx+w-dx, offy+h-dy)

        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
