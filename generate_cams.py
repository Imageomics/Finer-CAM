import os
import difflib
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pytorch_grad_cam import FinerCAM, GradCAM
from class_names import class_names_car
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class ModifiedDINO(nn.Module):
    """
    A wrapper for the original DINO model that adds a classifier layer.
    """
    def __init__(self, original_model, classifier_path, num_classes, feature_dim=768):
        super(ModifiedDINO, self).__init__()
        self.original_model = original_model
        self.classifier = nn.Linear(feature_dim, num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location=device)
        )
        self.blocks = self.original_model.blocks
        print("ModifiedDINO initialized")

    def forward(self, x):
        features = self.original_model.forward_features(x)["x_norm_patchtokens"]
        features = features.mean(dim=1)
        logits = self.classifier(features)
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


def preprocess(image, patch_size=14, max_size=1000):
    image = image.convert("RGB")
    width, height = image.size

    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        width = int(width * scale)
        height = int(height * scale)
        image = image.resize((width, height), Image.BICUBIC)

    new_height_pixels = int(np.ceil(height / patch_size) * patch_size)
    new_width_pixels = int(np.ceil(width / patch_size) * patch_size)

    transform = Compose([
        Resize((new_height_pixels, new_width_pixels), interpolation=Image.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                  std=[0.26862954, 0.26130258, 0.27577711]),
    ])

    image_tensor = transform(image).to(torch.float32)
    
    grid_height = new_height_pixels // patch_size
    grid_width = new_width_pixels // patch_size
    
    return image_tensor, grid_height, grid_width


def run_finer_cam_on_dataset(dataset_path, cam, preprocess, save_path, device):
    """
    Run FinerCAM on a dataset of images.
    """
    os.makedirs(save_path, exist_ok=True)

    if os.path.isdir(dataset_path):
        image_list = get_image_paths_from_folder(dataset_path)
    else:
        with open(dataset_path, 'r') as file:
            image_list = [line.strip() for line in file.readlines()]

    modes = ["Baseline", "Finer-Default", "Finer-Compare"]

    for img_path in tqdm(image_list):
        image_filename = os.path.basename(img_path)
        class_name = os.path.basename(os.path.dirname(img_path))
        base_name = os.path.splitext(image_filename)[0]
        new_filename = f"{class_name}_{base_name}.npy"

        image_pil = Image.open(img_path).convert('RGB')
        original_width, original_height = image_pil.size
    
        target_idx = get_true_label_idx(class_name, class_names_car)
        image_tensor, grid_height, grid_width = preprocess(image_pil)
        image_tensor = image_tensor.unsqueeze(0).to(device)

        results_by_mode = {}
        for mode in modes:
            # When alpha = 0, FinerCAM degrades to Baseline
            if mode == "Baseline":
                grayscale_cam, _, main_category, comparison_categories = cam(
                    input_tensor=image_tensor,
                    targets = None,
                    target_idx=target_idx,
                    H=grid_height,
                    W=grid_width,
                    alpha=0
                )
            elif mode == "Finer-Default":
            # Our default setting: compare with the three most similar categories
                grayscale_cam, _, main_category, comparison_categories = cam(
                    input_tensor=image_tensor,
                    targets = None,
                    target_idx=target_idx,
                    H=grid_height,
                    W=grid_width,
                    comparison_categories=[1,2,3]
                )
            elif mode == "Finer-Compare":
            # Compare only with the most similar category
                grayscale_cam, _, main_category, comparison_categories = cam(
                    input_tensor=image_tensor,
                    targets = None,
                    target_idx=target_idx,
                    H=grid_height,
                    W=grid_width,
                    comparison_categories=[1]
                )

            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam_highres = cv2.resize(grayscale_cam, (original_width, original_height))
            results_by_mode[mode] = {
                "highres": np.array([grayscale_cam_highres], dtype=np.float16),
                "main_category": main_category,
                "comparison_categories": comparison_categories
            }

        np.save(os.path.join(save_path, new_filename), results_by_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform Finer-CAM on a dataset')
    parser.add_argument('--classifier_path', type=str, required=True,
                        help='Path to the classifier model')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the validation set')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to save FinerCAM results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    original_model = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True
    ).to(device)
    
    num_classes = 196
    model = ModifiedDINO(original_model, args.classifier_path, num_classes)
    model = model.to(device)

    target_layers = [model.blocks[-1].norm1]
    cam = FinerCAM(model=model, target_layers=target_layers,
                   reshape_transform=reshape_transform, base_method= GradCAM)

    run_finer_cam_on_dataset(args.dataset_path, cam, preprocess, args.save_path, device)