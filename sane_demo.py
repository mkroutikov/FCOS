# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os
import torch
import torch.nn as nn
from torchvision import transforms as T
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.image_list import to_image_list
from fcos_model import FCOSModel, FCOSPostProcessor

import time


class COCODemo(object):

    def __init__(
        self,
        checkpoint,
        confidence_thresholds_for_classes,
        show_mask_heatmaps=False,
        masks_per_dim=2,
        min_image_size=224,
        device='cpu',
        input_to_bgr255=True,
        input_pixel_mean = (102.9801, 115.9465, 122.7717),
        input_pixel_std = [1., 1., 1.],
        size_divisibility=32,
        draw_masks = False,
        draw_keypoints = False,
        inference_th=0.05,
        pre_nms_top_n=1000,
        nms_th=0.6,
        fpn_post_nms_top_n=100,
    ):
        self.model = FCOSModel(num_classes=1)
        self.model.eval()

        self.box_selector = FCOSPostProcessor(
            pre_nms_thresh=inference_th,
            pre_nms_top_n=pre_nms_top_n,
            nms_thresh=nms_th,
            fpn_post_nms_top_n=fpn_post_nms_top_n,
            min_size=0,
            num_classes=2  # here we count background??? -MK
        )

        self.device = torch.device(device)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        from maskrcnn_benchmark.utils.checkpoint import Checkpointer
        checkpointer = Checkpointer(self.model, save_dir='')
        _ = checkpointer.load(checkpoint)

        self.transforms = self.build_transform(input_to_bgr255, input_pixel_mean, input_pixel_std)

        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_thresholds_for_classes = torch.tensor(confidence_thresholds_for_classes)
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim
        self.size_divisibility = size_divisibility
        self.draw_masks = draw_masks
        self.draw_keypoints = draw_keypoints

    def build_transform(self, convert_to_bgr255, pixel_mean, pixel_std):
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

        normalize_transform = T.Normalize(mean=pixel_mean, std=pixel_std)

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.draw_masks:
            result = self.overlay_mask(result, top_predictions)
        if self.draw_keypoints:
            result = self.overlay_keypoints(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        image_list = to_image_list(image, self.size_divisibility)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            locations, box_cls, box_regression, centerness = self.model(image_list)
            predictions = self.box_selector(
                locations, box_cls, box_regression,
                centerness, image_list.image_sizes
            )

        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        assert len(predictions) == 1
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels")
        thresholds = self.confidence_thresholds_for_classes[(labels - 1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field("keypoints")
        kps = keypoints.keypoints
        scores = keypoints.get_field("logits")
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))
        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[: len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros(
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]
        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
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

    demo_im_names = os.listdir(args.images_dir)

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        args.checkpoint,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=800,
    )

    for im_name in demo_im_names:
        img = cv2.imread(os.path.join(args.images_dir, im_name))
        if img is None:
            continue
        start_time = time.time()
        composite = coco_demo.run_on_opencv_image(img)
        print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
        cv2.imshow(im_name, composite)
    print("Press any keys to exit ...")
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

