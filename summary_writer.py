import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import cv2 as cv
import numpy as np
import torch.nn.functional as F


class TensorboardSummary(SummaryWriter):

    def visualize_image(self, image, target, output, global_step, title='Sample'):
        bitmap = image.clone().cpu().data
        truth  = decode_seg_map_sequence(torch.squeeze(target, 1).detach().cpu().numpy(), dataset=dataset)
        pred   = decode_seg_map_sequence(torch.max(output, 0)[1].detach().cpu().numpy(), dataset=dataset)

        sample = make_grid([bitmap, truth, pred], 3, normalize=True, scale_each=True)

        self.add_image(title, sample, global_step)

    def visualize_box(self, image, target=None, output=None, logits=None, global_step=None, title='Box_Sample'):
        bitmap = image.clone().cpu().data

        bitmap = bitmap.permute(1, 2, 0).numpy()
        height, width = bitmap.shape[:2]

        bmin = bitmap.min()
        bmax = bitmap.max()
        bitmap = (bitmap-bmin) * 255 / (bmax-bmin)
        bitmap = bitmap.astype(np.uint8)

        mask = None
        if bitmap.shape[-1] == 4:
            bitmap, mask = bitmap[:,:,:3], np.repeat(bitmap[:,:,3:], 3, axis=2)
        c = bitmap.copy()

        if target is not None:
            for x0,y0,x1,y1 in target.bbox:
                cv.rectangle(c, (int(x0), int(y0)), (int(x1), int(y1)), color=(0, 255, 0))

        if output is not None:
            for x0,y0,x1,y1 in output.bbox:
                cv.rectangle(c, (int(x0), int(y0)), (int(x1), int(y1)), color=(0, 0, 255))

        # cv.imshow('XXX', c)
        bitmap = torch.from_numpy(c).float().permute(2, 0, 1)
        grid = [bitmap]
        if mask is not None:
            mask = torch.from_numpy(mask).float().permute(2, 0, 1)
            grid.append(mask)
        if logits is not None:
            logits = logits.clone().cpu().data.unsqueeze(0)
            logits = F.interpolate(logits, size=bitmap.shape[1:])
            grid.append(logits[:, 0].repeat(3, 1, 1))
            grid.append(logits[:, 1].repeat(3, 1, 1))
            grid.append(logits[:, 2].repeat(3, 1, 1))
            grid.append(logits[:, 3].repeat(3, 1, 1))
        sample = make_grid(grid, len(grid), normalize=True, scale_each=True)

        self.add_image(title, sample, global_step)


if __name__ == '__main__':
    from sane_train import make_train_data_loader
    import logging

    logging.basicConfig(level=logging.INFO)

    dataloader = make_train_data_loader()

    summary = TensorboardSummary(logdir='testme')

    for imagelist, boxes, _ in dataloader:
        batch_size = len(imagelist.tensors)
        for jj in range(batch_size):
            img = imagelist.tensors[jj]
            boxlist = boxes[jj]

            summary.visualize_box(img, boxlist, None, jj+1)
        break

    cv.waitKey()


