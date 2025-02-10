"""
pt-cpg/google/vit-large-patch16-224
https://github.com/Pervasive-Technologies/pt-cpg-google-vit-large-patch16-224

| Copyright 2025, Pervasive Technologies.

|
"""
import torch
from transformers import ViTModel, ViTImageProcessor
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
    else:
        print("Model exist locally, not downloading.")

class ViTEmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure model weights are downloaded
        download_model()

        # Load the Vision Transformer model from Hugging Face (architecture only)
        self.model = ViTModel.from_pretrained(MODEL_PATH)
        
        # Load trained weights from model.pt
        #elf.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device).eval()

        # Use Hugging Face's preprocessing function
        self.processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

    def embed_image(self, image):
        """
        Extract embeddings from an image.
        - image: A PIL Image object
        - Returns: A normalized embedding vector (NumPy array)
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract embeddings from the CLS token (index 0)
        features = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        
        # Normalize embeddings
        norm = np.linalg.norm(features)
        return features / norm if norm > 0 else features

def load_model(model_name,model_path):
    """
    Entry point for FiftyOne Model Zoo.
    """
    return ViTEmbeddingModel()
