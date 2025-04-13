import numpy as np

class HelperFunctions:
    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    @staticmethod
    def non_max_suppression(boxes, scores, threshold):
        scores = np.array(scores)
        if len(boxes) == 0:
            return []
        indices = scores.argsort()[::-1]
        keep = []
        while len(indices) > 0:
            current_idx = indices[0]
            keep.append(current_idx)
            ious = [HelperFunctions.compute_iou(boxes[current_idx], boxes[idx]) for idx in indices[1:]]
            indices = [indices[i + 1] for i, iou in enumerate(ious) if iou < threshold]
        return keep