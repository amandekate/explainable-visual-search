"""
Offline embedding generation pipeline.

This script demonstrates how image embeddings are generated using CLIP
and indexed using FAISS. In practice, this pipeline was executed on
Google Colab with GPU support, and the generated artifacts were reused
locally by the FastAPI service.
"""

import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import faiss
from transformers import CLIPProcessor, CLIPModel


# -----------------------------
# Paths
# -----------------------------
IMAGE_DIR = "data/images"
EMBEDDING_DIR = "data/embeddings"
FAISS_DIR = "data/faiss_index"

EMBEDDINGS_PATH = os.path.join(EMBEDDING_DIR, "image_embeddings.npy")
IMAGE_PATHS_PATH = os.path.join(EMBEDDING_DIR, "image_paths.pkl")
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "image_index.faiss")


# -----------------------------
# Setup
# -----------------------------
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# -----------------------------
# Load images
# -----------------------------
image_paths = [
    os.path.join(IMAGE_DIR, img)
    for img in os.listdir(IMAGE_DIR)
    if img.lower().endswith((".jpg", ".png", ".jpeg"))
]

print(f"Found {len(image_paths)} images")


# -----------------------------
# Generate embeddings
# -----------------------------
embeddings = []

for img_path in tqdm(image_paths, desc="Generating embeddings"):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    embeddings.append(image_features.cpu().numpy())

embeddings = np.vstack(embeddings).astype("float32")


# -----------------------------
# Save embeddings
# -----------------------------
np.save(EMBEDDINGS_PATH, embeddings)

with open(IMAGE_PATHS_PATH, "wb") as f:
    pickle.dump(image_paths, f)

print("Embeddings saved")


# -----------------------------
# Build FAISS index
# -----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)

print("FAISS index saved")
