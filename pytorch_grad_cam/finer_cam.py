"""Finer-CAM wrapper built on top of a base CAM method.

This module adapts an existing CAM implementation, such as ``GradCAM``, to use
the Finer-CAM objective from https://arxiv.org/pdf/2501.11309. The wrapper
reuses the underlying activation/gradient collection logic and only changes how
the optimization target is formed before the per-layer CAM maps are computed.
"""

import numpy as np
import torch
from typing import List
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import FinerWeightedTarget

# Finer-CAM: https://arxiv.org/pdf/2501.11309

class FinerCAM:
    """Generate Finer-CAM saliency maps with an existing CAM backend.

    The class wraps a standard CAM implementation and replaces the target
    objective with :class:`~pytorch_grad_cam.utils.model_targets.FinerWeightedTarget`.
    When explicit ``targets`` are not provided, it derives a main category and a
    set of reference categories from the model outputs for each sample.
    """

    def __init__(self, model, target_layers, reshape_transform=None, base_method=GradCAM):
        """Initialize the Finer-CAM wrapper.

        Args:
            model: Model used to compute activations and gradients.
            target_layers: Layers from which CAM activations are collected.
            reshape_transform: Optional transform applied to layer outputs before
                CAM computation, typically used for transformer backbones.
            base_method: CAM class used for the underlying implementation. It
                must follow the :class:`~pytorch_grad_cam.base_cam.BaseCAM`
                interface. Defaults to :class:`~pytorch_grad_cam.GradCAM`.
        """
        self.base_cam = base_method(model, target_layers, reshape_transform)
        self.compute_input_gradient = self.base_cam.compute_input_gradient
        self.uses_gradients = self.base_cam.uses_gradients

    def __call__(self, *args, **kwargs):
        """Alias for :meth:`forward`."""
        return self.forward(*args, **kwargs)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module] = None,
                target_size = None,
                eigen_smooth: bool = False,
                alpha: float = 1,
                reference_category_ranks: List[int] = [1, 2, 3],
                target_idx: int = None,
                H: int = None,
                W: int = None
                ) -> np.ndarray:
        """Compute a Finer-CAM map for the given input batch.

        Args:
            input_tensor: Input batch passed to the model.
            targets: Optional list of `pytorch_grad_cam` target callables.
                Please refer to `~pytorch_grad_cam.utils.model_targets`.
                If ``None``, Finer-CAM targets are constructed automatically 
                based on the model outputs.
            target_size: Optional spatial output size passed to the wrapped CAM
                backend when resizing each per-layer map.
            eigen_smooth: Whether to apply eigenvalue-based smoothing in the
                wrapped CAM implementation.
            alpha: Scaling factor used in
                :class:`~pytorch_grad_cam.utils.model_targets.FinerWeightedTarget`
                for penalizing reference categories.
            reference_category_ranks: Indices into the sorted similarity list
                used to choose reference categories when ``targets`` is
                ``None``. Finer-CAM uses the second to fourth most similar
                categories as reference categories by default. If a requested
                rank exceeds the number of available classes, it is ignored.
            target_idx: The index of the target category. Usually the ground truth
                category. If omitted, the highest scoring category in each sample
                is used.
            H: Optional height argument forwarded to the activation/gradient
                extractor. This is used by some backbones that need explicit
                feature map sizing.
            W: Optional width argument forwarded to the activation/gradient
                extractor.

        Returns:
            A tuple ``(cam, outputs, main_categories, references)`` where:

            - ``cam`` is the aggregated CAM map from the wrapped backend.
            - ``outputs`` are the raw model outputs returned by the backend
              activation/gradient pass.
            - ``main_categories`` contains the automatically selected main
              category for each sample when ``targets`` is ``None``; otherwise
              it remains empty.
            - ``references`` contains the automatically selected reference
              categories for each sample when ``targets`` is ``None``;
              otherwise it remains empty.
        """

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.base_cam.activations_and_grads(input_tensor, H, W)

        main_categories = []
        references = []

        # Construct Finer-CAM targets if not provided.
        if targets is None:
            if isinstance(outputs, (list, tuple)):
                output_data = outputs[0].detach().cpu().numpy()
            else:
                output_data = outputs.detach().cpu().numpy()

            # Rank categories by absolute logit distance to the reference logit.
            # The closest category becomes the main category and the selected
            # ranks become the reference set used by FinerWeightedTarget.
            sorted_indices = np.empty_like(output_data, dtype=int)
            # Sort indices based on similarity to the target logit,
            # with more similar values (smaller differences) appearing first.
            for i in range(output_data.shape[0]):
                target_logit = output_data[i][np.argmax(output_data[i])] if target_idx is None else output_data[i][target_idx]
                differences = np.abs(output_data[i] - target_logit)
                sorted_indices[i] = np.argsort(differences)

            targets = []
            for i in range(sorted_indices.shape[0]):
                main_category = int(sorted_indices[i, 0])
                valid_reference_ranks = [
                    idx for idx in reference_category_ranks
                    if idx < sorted_indices.shape[1]
                ]
                current_reference = [
                    int(sorted_indices[i, idx]) for idx in valid_reference_ranks
                ]
                main_categories.append(main_category)
                references.append(current_reference)
                target = FinerWeightedTarget(main_category, current_reference, alpha)
                targets.append(target)

        if self.uses_gradients:
            self.base_cam.model.zero_grad()
            if isinstance(outputs, (list, tuple)):
                loss = sum([target(output) for target, output in zip(targets, outputs)])
            else:
                loss = sum([target(output) for target, output in zip(targets, [outputs])])
            loss.backward(retain_graph=True)

        cam_per_layer = self.base_cam.compute_cam_per_layer(
            input_tensor, targets, target_size, eigen_smooth
        )

        return self.base_cam.aggregate_multi_layers(cam_per_layer), outputs, main_categories, references
