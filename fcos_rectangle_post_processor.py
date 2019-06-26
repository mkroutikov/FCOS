import torch

from maskrcnn_benchmark.modeling.rpn.inference import RPNPostProcessor
from maskrcnn_benchmark.modeling.rpn.utils import permute_and_flatten

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import remove_small_boxes


class FCOSRectanglePostProcessor:
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def _process_one_level(self, locations, focus, regression, image_sizes):
        N, C, H, W = focus.shape
        assert C == 4

        # put in the same format as locations
        focus = focus.view(N, C, H, W).permute(0, 2, 3, 1).reshape(N, -1, C)
        regression = regression.view(N, C, H, W).permute(0, 2, 3, 1).reshape(N, -1, C)

        results = []
        for i in range(N):
            lc = torch.argmax(focus[i, :, 0])
            tc = torch.argmax(focus[i, :, 1])
            rc = torch.argmax(focus[i, :, 2])
            bc = torch.argmax(focus[i, :, 3])

            l = locations[lc, 0] - regression[i, lc, 0]
            t = locations[tc, 1] - regression[i, tc, 1]
            r = locations[rc, 0] + regression[i, rc, 2]
            b = locations[rc, 1] + regression[i, bc, 3]

            detections = torch.stack([l, t, r, b], dim=0).unsqueeze(0)
            c = focus[i, lc, 0] + focus[i, tc, 1] + focus[i, rc, 2] + focus[i, bc, 3]

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("scores", c.unsqueeze(0))
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
        locations, focus, regression = (logits[x] for x in ['locations', 'focus', 'regression'])
        sampled_boxes = []
        for location, foc, reg in zip(locations, focus, regression):
            sampled_boxes.append(
                self._process_one_level(
                    location, foc, reg, image_sizes
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
