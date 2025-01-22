import os
import difflib
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pytorch_grad_cam import GradCAM
from dinov2.models.vision_transformer import vit_base
from class_names import class_names_car

class ModifiedDINO(nn.Module):
    """
    A wrapper around the original DINO model that adds a classifier layer.
    This class initializes a linear layer for classification and loads its weights.
    """
    def __init__(self, original_model, classifier_path, num_classes):
        super(ModifiedDINO, self).__init__()
        self.original_model = original_model
        self.classifier = nn.Linear(768, num_classes)
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        )
        self.blocks = self.original_model.blocks
        print("ModifiedDINO initialized")

    def forward(self, x):
        image_features = original_model.forward_features(x)["x_norm_patchtokens"]
        image_features = image_features.mean(dim=1)
        logits = self.classifier(image_features)
        return logits

def get_image_paths_from_folder(folder_path):
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_true_label_idx(class_name, class_names):
    closest_match = difflib.get_close_matches(class_name, class_names, n=1, cutoff=0.8)
    if closest_match:
        return class_names.index(closest_match[0])
    return None

def preprocess(image):
    image = image.resize((224, 224), Image.BICUBIC)
    image = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    image = (image - mean) / std
    image = torch.tensor(image).permute(2, 0, 1)
    return image

def run_finer_cam_on_dataset(dataset_path, cam, preprocess, save_dir, device):
    """
      Run finer-cam on a dataset of images. 
    """
    os.makedirs(save_dir, exist_ok=True)

    if os.path.isdir(dataset_path):
        image_list = get_image_paths_from_folder(dataset_path)
    else:
        with open(dataset_path, 'r') as file:
            image_list = [line.strip() for line in file.readlines()]
    parameter_settings = {
        'w1': {'alpha': 0, 'n': 0, 'k': 1, 'x': 2, 'y': 3},
        'w1-w2': {'alpha': 1, 'n': 0, 'k': 1, 'x': 1, 'y': 1},
        'w1-w3': {'alpha': 1, 'n': 0, 'k': 2, 'x': 2, 'y': 2},
        'aggregate': {'alpha': 0.6, 'n': 0, 'k': 1, 'x': 2, 'y': 3},
    }

    for img_idx, img_path in enumerate(tqdm(image_list)):
        image_filename = os.path.basename(img_path)
        class_name = os.path.basename(os.path.dirname(img_path))

        # Construct a new filename for saving the results
        base_name = os.path.splitext(image_filename)[0]
        new_filename = f"{class_name}_{base_name}.npy"

        image_pil = Image.open(img_path).convert('RGB')
        true_label_idx = get_true_label_idx(class_name, class_names_car)
        image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

        results_by_config = {}
        for config_name, param_setting in parameter_settings.items():
            alpha = param_setting.get('alpha', 0.0)
            k = param_setting.get('k', None)
            n = param_setting.get('n', 1)

            # Prepare arguments for cam call
            cam_kwargs = {
                'input_tensor': image_tensor,
                'targets': None,
                'target_size': None,
                'alpha': alpha,
                'k': k,
                'n': n,
                'true_label_idx': true_label_idx,
            }

            # The cam call returns grayscale_cam, some placeholder, and class indices
            grayscale_cam, _, class_n_idx, class_k_idx = cam(**cam_kwargs)
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam_highres = cv2.resize(grayscale_cam, (224, 224))

            # Store results
            results_by_config[config_name] = {
                "highres": np.array([grayscale_cam_highres], dtype=np.float16),
                "class_n_idx": class_n_idx,
                "class_k_idx": class_k_idx
            }

        # Save all parameter setting results for this image
        np.save(os.path.join(save_dir, new_filename), results_by_config)

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Finer-CAM on a dataset')
    parser.add_argument('--classifier_path', type=str, required=True, help='Path to the classifier model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the valset')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save CAMs')
    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    original_model = vit_base(
        patch_size=14,
        img_size=518,
        init_values=1.0,
        block_chunks=0
    )

    original_model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    num_classes = 196
    model = ModifiedDINO(original_model, args.classifier_path, num_classes)
    model = model.to(args.device)

    target_layers = [model.blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    run_finer_cam_on_dataset(args.image_paths, cam, preprocess, args.save_dir, args.device)