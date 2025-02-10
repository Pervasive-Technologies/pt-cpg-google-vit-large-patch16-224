import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from google.cloud import storage
from google.oauth2 import service_account
import requests
import json

# Example usage

MODEL_PATH = "model.pt"

class ViTEmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"        
        # Load the model architecture (ViT-Large Patch16-224)
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


