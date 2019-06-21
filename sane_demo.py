# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms as T
import transforms_mask as TTT
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.image_list import to_image_list
from fcos_model import FCOSModel
from fcos_post_processor import FCOSPostProcessor
import contextlib
import time
import matplotlib.pyplot as plt


@contextlib.contextmanager
def timeit(title='Time elapsed'):
    start = time.time()
    try:
        yield
    finally:
        print('%s: %.2fs' % (title, (time.time()-start)))


def build_transform(convert_to_bgr255=True, pixel_mean=(102.9801, 115.9465, 122.7717), pixel_std=(1., 1., 1.)):
    """
    Creates a basic transformation that was used to train the models
    """

    transform = TTT.Compose(
        [
            TTT.PadToDivisibility(32),
            TTT.ToTensor(),
            TTT.Normalize(mean=pixel_mean, std=pixel_std, to_bgr255=convert_to_bgr255),
            TTT.MakeMaskChannel(),
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

    draw = ImageDraw.Draw(image)
    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        xyxy = box[0], box[1], box[2], box[3]
        draw.rectangle(xyxy, outline=(0, 255, 0))

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
    parser.add_argument(
        "--images-dir",
        default="demo/scarlet_images",
        metavar="DIR",
        help="path to demo images directory",
    )

    args = parser.parse_args()


    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
    thresholds_for_classes = [
        0.05,
        0.23860901594161987, 0.24108672142028809, 0.2470853328704834,
        0.2316885143518448, 0.2708061933517456, 0.23173952102661133,
        0.31990334391593933, 0.21302376687526703, 0.20151866972446442,
        0.20928964018821716, 0.3793887197971344, 0.2715213894844055,
        0.2836397588253021, 0.26449233293533325, 0.1728038638830185,
        0.314998596906662, 0.28575003147125244, 0.28987520933151245,
        0.2727000117301941, 0.23306897282600403, 0.265937477350235,
        0.32663893699645996, 0.27102580666542053, 0.29177549481391907,
        0.2043062448501587, 0.24331751465797424, 0.20752687752246857,
        0.22951272130012512, 0.22753854095935822, 0.2159966081380844,
        0.1993938684463501, 0.23676514625549316, 0.20982342958450317,
        0.18315598368644714, 0.2489681988954544, 0.24793922901153564,
        0.287187397480011, 0.23045086860656738, 0.2462811917066574,
        0.21191294491291046, 0.22845126688480377, 0.24365000426769257,
        0.22687821090221405, 0.18365581333637238, 0.2035856395959854,
        0.23478077352046967, 0.18431290984153748, 0.18184082210063934,
        0.2708037495613098, 0.2268175482749939, 0.19970566034317017,
        0.21832780539989471, 0.21120598912239075, 0.270445853471756,
        0.189377561211586, 0.2101106345653534, 0.2112293541431427,
        0.23484709858894348, 0.22701986134052277, 0.20732736587524414,
        0.1953316181898117, 0.3237660229206085, 0.3078872859477997,
        0.2881140112876892, 0.38746657967567444, 0.20038367807865143,
        0.28123822808265686, 0.2588447630405426, 0.2796839773654938,
        0.266757994890213, 0.3266656696796417, 0.25759157538414,
        0.2578003704547882, 0.17009201645851135, 0.29051828384399414,
        0.24002137780189514, 0.22378061711788177, 0.26134759187698364,
        0.1730124056339264, 0.1857597529888153
    ]

    model = FCOSModel(num_classes=1, num_input_channels=4)
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    import pdb; pdb.set_trace()
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

    transform = build_transform()

    count = 0
    for im_name in os.listdir(args.images_dir):
        image = Image.open(os.path.join(args.images_dir, im_name)).convert('RGB')
        if image is None:
            continue

        mask = Image.new('L', image.size)

        while True:
            with timeit('%s\tinference time' % im_name):
                (img, _, _), _ = transform(image, mask, None)
                img = img.unsqueeze(0)  # batch dimension

                height, width = img.shape[-2:]

                # compute predictions
                with torch.no_grad():
                    logits = model(img)
                    # always single image is passed at a time
                    prediction = box_selector(logits, [[height, width]])[0]

                scores = prediction.get_field("scores")
                labels = prediction.get_field("labels")
                thresholds = torch.tensor(thresholds_for_classes)[(labels - 1).long()]
                keep = torch.nonzero(scores > thresholds).squeeze(1)
                prediction = prediction[keep]
                scores = prediction.get_field("scores")
                _, idx = scores.sort(0, descending=True)
                top_predictions = prediction[idx]
                top_predictions = top_predictions[:1]
                print(top_predictions.bbox)

                composite = overlay_boxes(image.copy(), top_predictions)
                # composite = overlay_class_names(composite, top_predictions)

            plt.figure()
            plt.subplot(121)
            plt.imshow(composite)
            plt.subplot(122)
            plt.imshow(mask)
            count += 1

            if top_predictions.bbox.shape[0] == 0 or count > 2:
                break
            draw = ImageDraw.Draw(mask)
            xyxy = top_predictions.bbox[0].tolist()
            draw.rectangle(xyxy, fill=255)

        break

    plt.show()


if __name__ == "__main__":
    main()

