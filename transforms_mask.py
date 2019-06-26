from PIL import Image, ImageOps
import random
import torch
from torchvision.transforms import functional as F
from maskrcnn_benchmark.structures.image_list import to_image_list


class PadToDivisibility:
    '''Pads image (and target) to ensure height and width are divisible by |divisibility| '''

    def __init__(self, divisibility=32):
        self.divisibility = divisibility

    def __call__(self, *av, **kav):
        image, mask, target = av
        w, h = image.size

        d = self.divisibility
        height = ((h + d - 1) // d) * d
        width = ((w + d - 1) // d) * d

        l = (width - w) // 2
        r = width - w - l
        t = (height - h) // 2
        b = height - h - t

        image = ImageOps.expand(image, (l, t, r, b), fill='white')
        if mask is not None:
            mask  = ImageOps.expand(mask, (l, t, r, b), fill=0)
        if target is not None:
            target = target.pad(l, t, r, b)

        return (image, mask, target), {}



class RandomCrop:
    '''Randomly crops an image to a smaller one by |delta_x|, |delta_y|'''

    def __init__(self, delta_x=0, delta_y=0):
        self.delta_x = delta_x
        self.delta_y = delta_y

    def __call__(self, *av, **kav):
        image, mask, target = av
        dx = self.delta_x
        dy = self.delta_y

        if dx == 0 and dy == 0:
            return image, target

        offx = random.randint(0, dx)
        offy = random.randint(0, dy)

        w, h = image.size
        image = image.crop((offx, offy, offx+w-dx, offy+h-dy))
        mask  = mask.crop((offx, offy, offx+w-dx, offy+h-dy))
        if target is not None:
            target = target.crop((offx, offy, offx+w-dx, offy+h-dy))

        return (image, mask, target), {}


class Crop:
    '''crops an image by |delta_x|, |delta_y|'''

    def __init__(self, delta_x=0, delta_y=0):
        self.delta_x = delta_x
        self.delta_y = delta_y

    def __call__(self, *av, **kav):
        image, mask, target = av
        dx = self.delta_x
        dy = self.delta_y

        if dx == 0 and dy == 0:
            return image, target

        offx = dx // 2
        offy = dy // 2

        w, h = image.size
        image = image.crop((offx, offy, offx+w-dx, offy+h-dy))
        mask  = mask.crop((offx, offy, offx+w-dx, offy+h-dy))
        if target is not None:
            target = target.crop((offx, offy, offx+w-dx, offy+h-dy))

        return (image, mask, target), {}

class MakeMaskChannel:
    def __call__(self, *av, **kav):
        image, mask, target = av
        image = torch.cat([image, mask], dim=0)
        return (image, mask, target), {}

class ToTensor(object):
    def __call__(self, *av, **kav):
        image, mask, target = av
        return (F.to_tensor(image), F.to_tensor(mask), target), {}


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, *av, **kav):
        image, mask, target = av
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
            if mask is not None:
                mask = mask * 255 - 128.
        image = F.normalize(image, mean=self.mean, std=self.std)
        return (image, mask, target), {}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *av, **kav):
        for t in self.transforms:
            av, kav = t(*av, **kav)
        return av, kav

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class BatchCollator:
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """
    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0])
        masks  = to_image_list(transposed_batch[1])
        targets = transposed_batch[2]
        img_ids = transposed_batch[3]
        return images, masks, targets, img_ids
