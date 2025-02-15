import numpy as np
import torch
import torchvision


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


class ClassifierOutputSoftmaxTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.softmax(model_output, dim=-1)[self.category]
        return torch.softmax(model_output, dim=-1)[:, self.category]


class BinaryClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if self.category == 1:
            sign = 1
        else:
            sign = -1
        return model_output * sign


class SoftmaxOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return torch.softmax(model_output, dim=-1)


class RawScoresOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return model_output


class SemanticSegmentationTarget:
    """ Gets a binary spatial mask and a category,
        And return the sum of the category scores,
        of the pixels in the mask. """

    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        if torch.backends.mps.is_available():
            self.mask = self.mask.to("mps")

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()
        elif torch.backends.mps.is_available():
            output = output.to("mps")

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()
            elif torch.backends.mps.is_available():
                box = box.to("mps")

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output

class DiffTarget:
    def __init__(
        self, 
        class_n_idx: int = 0, 
        class_k_idx: int = 1, 
        class_x_idx: int = 2, 
        class_y_idx: int = 3, 
        alpha: float = 1.0, 
        mode: str = "default", 
        single_target: int = 1 
    ):
        """
        mode="default"  -> Average aggregate 
        mode="weighted"  -> Post softmax weighted
        mode="single"    -> Direct comparison: wn - w[single_target]
        mode="baseline"  -> Baseline method
        """
        self.class_n_idx = class_n_idx
        self.class_k_idx = class_k_idx
        self.class_x_idx = class_x_idx
        self.class_y_idx = class_y_idx
        self.alpha = alpha
        self.mode = mode
        self.single_target = single_target   

#["GradCAM", "Finer-Default", "Finer-Weighted", "Finer-Compare"]

    def __call__(self, model_output):
        wn = model_output[..., self.class_n_idx]
        w_index = model_output[..., self.single_target]  
        
        if self.mode == "Finer-Default":
            numerator = (wn - self.alpha * model_output[..., self.class_k_idx]) + \
                        (wn - self.alpha * model_output[..., self.class_x_idx]) + \
                        (wn - self.alpha * model_output[..., self.class_y_idx])
            return numerator / 3

        elif self.mode == "Finer-Weighted":
            prob = torch.softmax(model_output, dim=-1)

            p_k = prob[..., self.class_k_idx]
            p_x = prob[..., self.class_x_idx]
            p_y = prob[..., self.class_y_idx]

            numerator = p_k * (wn - model_output[..., self.class_k_idx]) + \
                        p_x * (wn - model_output[..., self.class_x_idx]) + \
                        p_y * (wn - model_output[..., self.class_y_idx])
            denominator = p_k + p_x + p_y

            return numerator / (denominator + 1e-9)

        elif self.mode == "Finer-Compare":
            return wn - w_index  

        elif self.mode == "Baseline":
            return wn  

        else:
            raise ValueError("Invalid mode. Choose 'Finer-Default', 'Finer-Weighted', 'Finer-Compare', or 'Baseline'.")