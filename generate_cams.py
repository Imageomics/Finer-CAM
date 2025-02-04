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
from dinov2.models.vision_transformer import vit_base
from class_names import class_names_car


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
        # Use self.original_model to get features.
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


def preprocess(image):
    image = image.resize((224, 224), Image.BICUBIC)
    image_np = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    image_np = (image_np - mean) / std
    image_tensor = torch.tensor(image_np).permute(2, 0, 1)
    return image_tensor


def run_finer_cam_on_dataset(dataset_path, cam, preprocess_fn, save_dir, device):
    """
    Run FinerCAM on a dataset of images.
    """
    os.makedirs(save_dir, exist_ok=True)

    if os.path.isdir(dataset_path):
        image_list = get_image_paths_from_folder(dataset_path)
    else:
        with open(dataset_path, 'r') as file:
            image_list = [line.strip() for line in file.readlines()]

    modes = ["baseline", "default", "weighted", "single"]

    for img_path in tqdm(image_list):
        image_filename = os.path.basename(img_path)
        class_name = os.path.basename(os.path.dirname(img_path))
        base_name = os.path.splitext(image_filename)[0]
        new_filename = f"{class_name}_{base_name}.npy"

        try:
            image_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening {img_path}: {e}")
            continue

        true_label_idx = get_true_label_idx(class_name, class_names_car)
        image_tensor = preprocess_fn(image_pil).unsqueeze(0).to(device)

        results_by_mode = {}
        for mode in modes:
            grayscale_cam, _, class_n_idx, class_k_idx = cam.forward(
                input_tensor=image_tensor,
                targets=None,
                target_size=None,
                true_label_idx=true_label_idx,
                mode=mode
            )
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam_highres = cv2.resize(grayscale_cam, (224, 224))
            results_by_mode[mode] = {
                "highres": np.array([grayscale_cam_highres], dtype=np.float16),
                "class_n_idx": class_n_idx,
                "class_k_idx": class_k_idx
            }

        np.save(os.path.join(save_dir, new_filename), results_by_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform GradCAM on a dataset')
    parser.add_argument('--classifier_path', type=str, required=True,
                        help='Path to the classifier model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pretrained model')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to the validation set')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save FinerCAM results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    original_model = vit_base(
        patch_size=14,
        img_size=518,
        init_values=1.0,
        block_chunks=0
    )
    original_model.load_state_dict(torch.load(args.model_path, map_location=device))
    num_classes = 196
    model = ModifiedDINO(original_model, args.classifier_path, num_classes)
    model = model.to(device)

    target_layers = [model.blocks[-1].norm1]
    cam = FinerCAM(GradCAM, model=model, target_layers=target_layers,
                   reshape_transform=reshape_transform)

    run_finer_cam_on_dataset(args.dataset_dir, cam, preprocess, args.save_dir, device)