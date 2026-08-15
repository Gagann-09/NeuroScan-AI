# ai_pipeline/preprocessing/spatial_2d/kaggle_prep.py
from PIL import Image
import torch
from torchvision import transforms

def preprocess_image(local_file_path: str) -> torch.Tensor:
    """
    Loads a 2D image (.jpg, .jpeg, .png), resizes to 224x224, 
    applies standard ImageNet normalization, and converts to a PyTorch tensor.
    """
    img = Image.open(local_file_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(img)
    return tensor