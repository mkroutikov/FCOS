# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os
import torch
import torch.nn as nn
from torchvision import transforms as T
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.image_list import to_image_list
from fcos_model import FCOSModel
from fcos_post_processor import FCOSPostProcessor
import contextlib
import time
import torchsummary


@contextlib.contextmanager
def timeit(title='Time elapsed'):
    start = time.time()
    try:
        yield
    finally:
        print('%s: %.2fs' % (title, (time.time()-start)))


def build_transform(convert_to_bgr255=True, pixel_mean=(102.9801, 115.9465, 122.7717), pixel_std=(1., 1., 1.), min_image_size=224):
    """
    Creates a basic transformation that was used to train the models
    """

    # we are loading images with OpenCV, so we don't need to convert them
    # to BGR, they are already! So all we need to do is to normalize
    # by 255 if we want to convert to BGR255 format, or flip the channels
    # if we want it to be in RGB in [0-1] range.
    if convert_to_bgr255:
        to_bgr_transform = T.Lambda(lambda x: x * 255)
    else:
        to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(min_image_size),
            T.ToTensor(),
            to_bgr_transform,
            T.Normalize(mean=pixel_mean, std=pixel_std),
        ]
    )
    return transform


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    labels = predictions.get_field("labels")
    boxes = predictions.bbox

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image

def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    labels = ['block' for i in labels]
    boxes = predictions.bbox

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 3
        )
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 128), 1
        )

    return image


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--checkpoint",
        '-c',
        required=True,
        # default="FCOS_R_50_FPN_1x.pth",
        metavar="FILE",
        help="path to the trained model",
    )

    args = parser.parse_args()

    state_dict = torch.load(args.checkpoint, map_location='cpu')
    model = FCOSModel(num_classes=1)
    model.load_state_dict(state_dict['model'])
    model.eval()

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=0.05,
        pre_nms_top_n=1000,
        nms_thresh=0.6,
        fpn_post_nms_top_n=100,
        min_size=0,
        num_classes=2  # here we count background??? -MK
    )

    t = torch.zeros(3, 1056, 800)
#    t[0, 512+28, 400] = 100
    image_list = to_image_list([t])

    # compute predictions
    with torch.no_grad():
        prediction = model(image_list.tensors)
        for key in ['centerness', 'box_cls', 'box_regression']:
            logits = [x.squeeze(0) for x in prediction[key]]
            logits = [x.max(dim=0)[0] for x in logits]
            logits = [x.numpy() for x in logits]

            for x in range(len(logits)):
                cv2.imshow('sample-%s-%d %r' % (key, x, logits[x].shape), logits[x])
                cv2.imwrite('sample-%s-%d.png' % (key, x), logits[x])

    print("Press any keys to exit ...")
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

