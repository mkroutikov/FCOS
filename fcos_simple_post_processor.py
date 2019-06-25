import torch

from maskrcnn_benchmark.modeling.rpn.inference import RPNPostProcessor
from maskrcnn_benchmark.modeling.rpn.utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes


class FCOSSimplePostProcessor:
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def _process_one_level(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C)
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1)
        box_cls += centerness[:,:,None]
        box_cls = box_cls.sigmoid()

        # multiply the classification scores with centerness scores
        #box_cls = box_cls * centerness[:, :, None]

        best_boxes = torch.argmax(box_cls.view(N, -1), dim=1)
        best_box_cls = box_cls.view(N, -1)[:, best_boxes]

        results = []
        for i in range(N):
            c = best_box_cls[i]
            r = box_regression[i, best_boxes[i], :]
            l = locations[best_boxes[i],:]

            detections = torch.stack([
                l[0] - r[0],
                l[1] - r[1],
                l[0] + r[2],
                l[1] + r[3],
            ], dim=0).unsqueeze(0)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("scores", c)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            results.append(boxlist)

        return results

    def __call__(self, logits, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        locations, box_cls, box_regression, centerness = (logits[x] for x in ['locations', 'box_cls', 'box_regression', 'centerness'])
        sampled_boxes = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            sampled_boxes.append(
                self._process_one_level(
                    l, o, b, c, image_sizes
                )
            )

        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            scores = boxlists[i].get_field("scores")
            boxes = boxlists[i].bbox
            boxlist = boxlists[i]

            best_index = torch.argmax(scores, dim=0, keepdim=True)
            results.append(boxlist[best_index])
        return results
