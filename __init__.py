"""
pt-cpg/google/vit-large-patch16-224
https://github.com/Pervasive-Technologies/pt-cpg-google-vit-large-patch16-224

| Copyright 2025, Pervasive Technologies.

|
"""
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import requests
import json
import logging
import os

# Example usage

MODEL_PATH = "./model.pt"
MODEL_URL = "https://storage.googleapis.com/fiftyone-models/pt-cpg-google-vit-large-patch16-224/model.pt"  # Replace with your actual URL
MODEL_NAME = "pt-cpg/google/vit-large-patch16-224"

def download_model():
    """Downloads the model.

    Args:
        model_name: the name of the model to download, as declared by the
            ``base_name`` and optional ``version`` fields of the manifest
        model_path: the absolute filename or directory to which to download the
            model, as declared by the ``base_filename`` field of the manifest
    """
    
    """
    Downloads model.pt if it does not exist locally.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download complete.")


class ViTEmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        # Load the model architecture (ViT-Large Patch16-224)
        download_model()
        
        self.model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)  # No classifier head
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device).eval()
        
        # Define preprocessing pipeline (must match training setup)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to match ViT input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalization
        ])

    def embed_image(self, image):
            """
            Extracts embeddings from an image and normalizes them.
            
            Args:
                image (PIL.Image): Input image
            
            Returns:
                np.ndarray: Normalized feature embedding (1D array)
            """
            # Apply preprocessing
            image = self.transform(image).unsqueeze(0).to(self.device)

            # Extract embeddings
            with torch.no_grad():
                embedding = self.model(image)  # Shape: (1, feature_dim)

            # Convert to NumPy and normalize
            features = embedding.cpu().numpy().squeeze()  # Convert to (feature_dim,)
            norm = np.linalg.norm(features)

            if norm > 0:
                features = features / norm  # Normalize only if norm > 0

            return features

def load_model():
    """
    Entry point for FiftyOne Model Zoo.
    This function will be called to load the model.
    """
    return ViTEmbeddingModel()