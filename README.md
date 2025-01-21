
# Finer-CAM : Spotting the Difference Reveals Finer Details for Visual Explanation
Official implementation of "Finer-CAM  [[arxiv]](link)". 

CAM methods highlight image regions influencing predictions but often struggle in fine-grained tasks due to shared feature activation across similar classes. We propose **Finer-CAM**, which explicitly compares the target class with similar ones, suppressing shared features and emphasizing unique, discriminative details.

Finer-CAM retains CAM’s efficiency, offers precise localization, and adapts to multi-modal zero-shot models, accurately activating object parts or attributes. It enhances explainability in fine-grained tasks without increasing complexity.

![images](pipeline.jpg)

## Demo 
Experience the power of Finer-CAM with our interactive demos! Witness **accurate localization** of discriminative features.

- 	Try the **multi-modal** demo and see how Finer-CAM activates detailed and relevant regions for diverse concepts: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1plLrL7vszVD5r71RGX3YOEXEBmITkT90?usp=sharing)
- Test the **classifier** demo to explore class-specific activation maps with enhanced explainability: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SAjRTmGy31G-GjtAc9pVH6isPjm1hWsj?usp=sharing)

## Reqirements

```
# create conda env
conda create -n finer-cam python=3.9 -y
conda activate finer-cam

# install packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python ftfy regex tqdm ttach tensorboard lxml cython scikit-learn matplotlib
```




## Preparing Datasets
### Stanford Cars
1. **Download the dataset** using the following command:

   ```bash
   curl -L -o datasets/stanford_cars.zip \
   https://www.kaggle.com/api/v1/datasets/download/cyizhuo/stanford-cars-by-classes-folder


2. **Unzip the downloaded file** 
   ```bash
   unzip datasets/stanford_cars.zip -d datasets/

3. The structure of `/your_home_dir/datasets/`should be organized as follows:

```
datasets/
├── train/
│   ├── Acura Integra Type R 2001/
│   │   ├── 000405.jpg
│   │   ├── 000406.jpg
│   │   └── ...
│   ├── Acura RL Sedan 2012/
│   │   ├── 000090.jpg
│   │   ├── 000091.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── Acura Integra Type R 2001/
    │   ├── 000450.jpg
    │   ├── 000451.jpg
    │   └── ...
    ├── Acura RL Sedan 2012/
    │   ├── 000122.jpg
```

### Preparing pre-trained model
Download DINOv2 pre-trained [ViT-B/14] at [here](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) and put it to `/your_home_dir/pretrained_models/dinov2`.

## Usage

### Step 1. Generate CAMs for Validation Set

 **Run the Script:**

   - Execute the `generate_cam.py` script with the appropriate arguments using the following command:
     ```bash
     python generate_cam.py --save_dir <path_to_save_results> \
                            --classifier_path <path_to_classifier_model> \
                            --model_path <path_to_dino_model> \
                            --image_paths <path_to_dataset_or_image_list>
     ```




### Step 2. Visualize Results

 **Run the Script:**

   - Execute the `visualize.py` script with the appropriate arguments using the following command:
     ```bash
     python visualize.py --dataset_dir <path_to_dataset_directory> \
                         --cam_dir <path_to_cam_directory> \
                         --save_dir <path_to_save_visualizations>
     ```





## Acknowledgement

We utilized code from:

- [dinov2](https://github.com/facebookresearch/dinov2?tab=readme-ov-file)  
- [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam/tree/61e9babae8600351b02b6e90864e4807f44f2d4a)  
- [clip-es](https://github.com/linyq2117/CLIP-ES)  
- [clip](https://github.com/openai/CLIP)

  Thanks for their wonderful works.


# Citation
If you find this repository useful, please consider citing our work :pencil: and giving a star :star2: :
```
@article{
}
```