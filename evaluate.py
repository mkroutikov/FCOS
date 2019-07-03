import utils
import torch
import time
import collections


@torch.no_grad()
def evaluate(model, data_loader, device, print_every=1):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    ilabs_evaluator = IlabsEvaluator()

    for image, targets in metric_logger.log_every(data_loader, print_every, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        evaluator_time = time.time()
        ilabs_evaluator.update(outputs, targets)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # accumulate predictions from all images
    ilabs_evaluator.summarize()

    return ilabs_evaluator


class IlabsEvaluator:

    def __init__(self):
        self._stats = collections.defaultdict(int)
        self._tp_scores = []
        self._fp_scores = []

    def update(self, outputs, targets):

        for prediction, gt in zip(outputs, targets):
            boxes = prediction["boxes"]
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            assert all(l==1 for l in labels)  # FIXME: extend to handle multiple classes -MK

            predicted = [
                {
                    'id'   : k,
                    'label': labels[k],
                    'score': scores[k],
                    'bbox' : box
                }
                for k, box in enumerate(boxes)
            ]

            self._update(predicted, gt)

    def _update(self, predicted, target):
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        assigned = {}
        for gt_box, gt_label in zip(gt_boxes, gt_labels):
            assert gt_label == 1  # FIXME: extend to handle multiple classes -MK
            # greedily find the best-matching prediction

            candidates = [{
                    'id': p['id'],
                    'iou': iou(gt_box, p['bbox']),
                    'score': p['score'],
                    'bbox' : p['bbox'],
                }
                for p in predicted if p['id'] not in assigned
            ]

            if not candidates:
                break

            if all(p['iou'] == 0. for p in candidates):
                break

            candidates = sorted(candidates, key=lambda x: x['iou'])
            best = candidates[-1]
            best['gt_bbox'] = gt_box
            assigned[best['id']] = best

        tp = len(assigned)
        fn = len(gt_boxes) - len(assigned)  # these boxes were not found by predictor (False Negatives)
        fp = len(predicted) - len(assigned)  # these boxes were predicted, but not matched (False positives)
        miou = sum(p['iou'] for p in assigned.values()) / (len(assigned) + 1.e-7)  # mean iou - how precise our tp boxes are

        self._stats['tp'] += tp
        self._stats['fn'] += fn
        self._stats['fp'] += fp
        self._stats['miou'] += miou
        self._stats['count'] += 1
        self._stats['total_gt'] += len(gt_boxes)
        self._stats['total_predicted'] += len(predicted)

        self._tp_scores.extend(p['score'] for p in predicted if p['id'] in assigned)
        self._fp_scores.extend(p['score'] for p in predicted if p['id'] not in assigned)

    def summarize(self):
        for k in sorted(self._stats.keys()):
            print(k, '\t', self._stats[k])

        mean_tp_score = sum(self._tp_scores) / (len(self._tp_scores) + 1e-7)
        mean_fp_score = sum(self._fp_scores) / (len(self._fp_scores) + 1e-7)
        print('mean_tp_score', mean_tp_score)
        print('mean_fp_score', mean_fp_score)


def area(l, t, r, b):
    return (r-l) * (b-t)


def iou(t1, t2):
    x0 = max(t1[0], t2[0])
    x1 = min(t1[2], t2[2])

    w = max(0, x1-x0)

    y0 = max(t1[1], t2[1])
    y1 = min(t1[3], t2[3])

    h = max(0, y1-y0)

    area_intersection = h * w

    area_union = area(t1[0], t1[1], t1[2], t1[3]) + area(t2[0], t2[1], t2[2], t2[3]) - area_intersection

    return area_intersection / (area_union + 1e-7)

