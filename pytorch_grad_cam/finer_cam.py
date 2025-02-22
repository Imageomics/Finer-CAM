import numpy as np
import torch
from typing import List
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import DiffTarget

# Finer-CAM: https://arxiv.org/pdf/2501.11309

class FinerCAM:
    def __init__(self, base_method, model, target_layers, reshape_transform=None):
        assert issubclass(base_method, BaseCAM), "base_method must inherit from BaseCAM"
        self.base_cam = base_method(model, target_layers, reshape_transform)

        self.compute_input_gradient = self.base_cam.compute_input_gradient
        self.uses_gradients = self.base_cam.uses_gradients

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[DiffTarget] = None,
                target_size=None,
                eigen_smooth: bool = False,
                alpha: float = 0.6,
                n: int = 0,
                k: int = 1,
                x: int = 2,
                y: int = 3,
                true_label_idx: int = None,  
                mode: str = 'Finer-Default',
                H: int = None,
                W: int = None
                ) -> np.ndarray:

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        outputs = self.base_cam.activations_and_grads(input_tensor, H, W)

        if targets is None:
            if isinstance(outputs, (list, tuple)):
                output_data = outputs[0].detach().cpu().numpy()
            else:
                output_data = outputs.detach().cpu().numpy()

            sorted_indices = np.argsort(-output_data, axis=-1)
            for i in range(sorted_indices.shape[0]):
                current_sorted_indices = np.delete(sorted_indices[i], np.where(sorted_indices[i] == true_label_idx))
                sorted_indices[i] = np.insert(current_sorted_indices, 0, true_label_idx)
            
            targets = []
            for i in range(sorted_indices.shape[0]):
                class_n_idx = int(sorted_indices[i, n])
                class_k_idx = int(sorted_indices[i, k]) if k is not None else None
                class_x_idx = int(sorted_indices[i, x]) if x is not None else None
                class_y_idx = int(sorted_indices[i, y]) if y is not None else None

                target = DiffTarget(class_n_idx, class_k_idx, class_x_idx, class_y_idx, alpha, mode)
                targets.append(target)

        if self.uses_gradients:
            self.base_cam.model.zero_grad()
            if isinstance(outputs, (list, tuple)):
                loss = sum([target(output) for target, output in zip(targets, outputs)])
            else:
                loss = sum([target(output) for target, output in zip(targets, [outputs])])
            loss.backward(retain_graph=True)

        cam_per_layer = self.base_cam.compute_cam_per_layer(input_tensor,
                                                            targets,
                                                            target_size,
                                                            eigen_smooth)
        return self.base_cam.aggregate_multi_layers(cam_per_layer), outputs, class_n_idx, class_k_idx