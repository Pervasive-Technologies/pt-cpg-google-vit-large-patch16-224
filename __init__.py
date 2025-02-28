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
import fiftyone.core.models as fom

# Example usage

MODEL_PATH = "./"
MODEL_FILENAME = "model.safetensors"
MODEL_URL = "https://storage.googleapis.com/fiftyone-models/pt-cpg-google-vit-large-patch16-224/model.safetensors"  # Replace with your actual URL
MODEL_NAME = "pt-cpg-google-vit-large-patch16-224"
import os

def download_model(model_name,model_path):
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
    print("model loading:",model_path,os.getcwd())
    if not os.path.exists(model_path):
        print(f"Downloading model from {MODEL_URL}...")
        response = requests.get(MODEL_URL, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    else:
        print("Model exist locally!!!!, not downloading.")


class ViTEmbeddingModel(fom.Model):
    def __init__(self,model_name,model_path):
        super().__init__()  # Ensure compatibility with FiftyOne

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure model weights are downloaded
        download_model(model_name,model_path)

        # Load the Vision Transformer model from Hugging Face (architecture only)
        self.model = ViTModel.from_pretrained(os.path.dirname(model_path))
        
        # Load trained weights from model.pt
        #self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        #self.model.to(self.device).eval()

        # Use Hugging Face's preprocessing function
        self.processor = ViTImageProcessor.from_pretrained(os.path.dirname(model_path))

    def embed(self, image):
        """
        Extracts embeddings from an image.
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
        if norm > 0:
            return (features / norm).tolist()  
        else:
            return features.tolist() 

    @property
    def has_embeddings(self):
        """Ensures that the model exposes embeddings."""
        return True
    
    @property
    def media_type(self):
        """Defines the media type for FiftyOne (images in this case)."""
        return "image"  # Must be "image" or "video"


def load_model(model_name,model_path):
    """
    Entry point for FiftyOne Model Zoo.
    """
    return ViTEmbeddingModel(model_name,model_path)
