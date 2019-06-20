from PIL import Image, ImageOps
import random


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
            target = target.pad(l, t, r, b)

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
            target = target.crop((offx, offy, offx+w-dx, offy+h-dy))

        return image, target
