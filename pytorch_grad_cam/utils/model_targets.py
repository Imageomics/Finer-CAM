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

class FinerWeightedTarget:
    """Target callable implementing the Finer-CAM weighting objective.

    The target keeps a main category score while suppressing a set of
    reference categories. Each reference term is weighted by a strength
    parameter alpha, then normalized by the total weight. This produces
    a relative objective that emphasizes fine-grained discrimination
    between visually similar classes.
    """

    def __init__(self, main_category, reference_categories, alpha):
        """Store the categories used by the Finer-CAM objective.

        Args:
            main_category: Index of the target category to highlight.
            reference_categories: Category indices used as references to
                suppress competing evidence in the attribution objective.
            alpha: Scaling factor applied to each reference-category score
                before subtracting it from the main-category score.
        """
        self.main_category = main_category
        self.reference_categories = reference_categories
        self.alpha = alpha
    
    def __call__(self, model_output):
        """Evaluate the weighted Finer-CAM target on model logits.

        Args:
            model_output: A 1D tensor of class logits for a single sample or a
                tensor whose last dimension indexes classes.

        Returns:
            A scalar-like tensor representing the weighted relative score
            between the main category and the reference categories:

            ``sum_i p_i * (w_n - alpha * w_i) / (sum_i p_i + 1e-9)``

            where ``w_n`` is the main-category logit, ``w_i`` are the
            reference-category logits, and ``p_i`` are softmax probabilities
            of the reference categories. If a reference category index exceeds
            the number of available classes, it is ignored. If no valid
            reference categories remain, the target falls back to the
            main-category score.
        """
        select = lambda idx: model_output[idx] if len(model_output.shape) == 1 else model_output[..., idx]

        wn = select(self.main_category)
        num_classes = model_output.shape[0] if len(model_output.shape) == 1 else model_output.shape[-1]
        valid_reference_categories = [
            idx for idx in self.reference_categories if idx < num_classes
        ]

        if len(model_output.shape) == 1:
            prob = torch.softmax(model_output.unsqueeze(0), dim=-1).squeeze(0)
        else:
            prob = torch.softmax(model_output, dim=-1)

        if not valid_reference_categories:
            return wn

        weights = [
            prob[idx] if len(model_output.shape) == 1 else prob[..., idx]
            for idx in valid_reference_categories
        ]
        # Normalize the weighted margin so the target stays comparable across
        # different reference sets and confidence distributions.
        numerator = sum(
            w * (wn - self.alpha * select(idx))
            for w, idx in zip(weights, valid_reference_categories)
        )
        denominator = sum(weights)
        return numerator / (denominator + 1e-9)
