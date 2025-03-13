import numpy as np
import torch
from typing import List
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import FinerWeightedTarget
# Finer-CAM: https://arxiv.org/pdf/2501.11309

class FinerCAM:
    def __init__(self, base_method, model, target_layers, reshape_transform=None):
        assert issubclass(base_method, BaseCAM), "Initialize FinerCAM with a given CAM method."
        self.base_cam = base_method(model, target_layers, reshape_transform)
        self.compute_input_gradient = self.base_cam.compute_input_gradient
        self.uses_gradients = self.base_cam.uses_gradients

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                target_size=None,
                eigen_smooth: bool = False,
                alpha: float = 1,
                comparison_categories: List[int] = [1, 2, 3],
                target_idx: int = None,
                H: int = None,
                W: int = None
                ) -> np.ndarray:

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.base_cam.activations_and_grads(input_tensor, H, W)

        main_categories = []
        comparisons = []

        if targets is None:
            if isinstance(outputs, (list, tuple)):
                output_data = outputs[0].detach().cpu().numpy()
            else:
                output_data = outputs.detach().cpu().numpy()

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
                current_comparison = [int(sorted_indices[i, idx]) for idx in comparison_categories]
                main_categories.append(main_category)
                comparisons.append(current_comparison)
                target = FinerWeightedTarget(main_category, current_comparison, alpha)
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

        return self.base_cam.aggregate_multi_layers(cam_per_layer), outputs, main_categories, comparisons