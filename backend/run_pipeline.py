"""
Offline embedding generation pipeline.

This script generates image embeddings using OpenAI CLIP and indexes them
using FAISS. The pipeline is intended to be executed on Google Colab
with GPU support, and the generated artifacts are reused locally by
the FastAPI service.
"""

import os
import pickle
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from io import BytesIO

import torch
import clip
import faiss



# =========================
# Paths
# =========================

CSV_PATH = "photos_url.csv"   # input dataset (URLs)
EMBEDDING_DIR = "data/embeddings"
FAISS_DIR = "data/faiss_index"

IMAGE_URLS_PATH = os.path.join(EMBEDDING_DIR, "image_urls.pkl")
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "image_index.faiss")

os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)



# =========================
# Setup
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()



# =========================
# Load & sample dataset
# =========================

MAX_IMAGES = 2000   # adjust if needed

df = pd.read_csv(CSV_PATH)
image_urls = df["url"].dropna().sample(
    n=MAX_IMAGES,
    random_state=42
).tolist()

print(f"Using {len(image_urls)} images")



# =========================
# Generate embeddings
# =========================

embeddings = []

for url in tqdm(image_urls, desc="Generating embeddings"):
    try:
        response = requests.get(url, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embeddings.append(image_features.cpu().numpy())

    except Exception:
        # Skip broken images
        continue

embeddings = np.vstack(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)



# =========================
# Save metadata
# =========================

with open(IMAGE_URLS_PATH, "wb") as f:
    pickle.dump(image_urls[: embeddings.shape[0]], f)

print("Saved image_urls.pkl")



# =========================
# Build FAISS index
# =========================

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)

print("FAISS index saved")
