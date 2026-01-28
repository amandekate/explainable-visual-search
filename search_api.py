import pickle
import numpy as np
import pandas as pd

import torch
import faiss
from fastapi import FastAPI, Query
from transformers import CLIPProcessor, CLIPModel


# -----------------------------
# Paths
# -----------------------------
EMBEDDINGS_PATH = "data/embeddings/image_embeddings.npy"
IMAGE_PATHS_PATH = "data/embeddings/image_paths.pkl"
FAISS_INDEX_PATH = "data/faiss_index/image_index.faiss"
URLS_PATH = "data/embeddings/sample_urls.csv"


# -----------------------------
# Load FAISS + Metadata
# -----------------------------
image_embeddings = np.load(EMBEDDINGS_PATH)
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

with open(IMAGE_PATHS_PATH, "rb") as f:
    image_paths = pickle.load(f)
    
urls_df = pd.read_csv(URLS_PATH)

image_urls = urls_df["photo_image_url"].tolist()

print("Loaded embeddings:", image_embeddings.shape)
print("Loaded FAISS index with", faiss_index.ntotal, "vectors")


# -----------------------------
# Load CLIP (Text Encoder)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    ignore_mismatched_sizes=True
).to(device)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Multimodal Image Search API")

@app.get("/health")
def health():
    return {"status": "ok"}



# -----------------------------
# Search Endpoint
# -----------------------------
@app.get("/search")
def search_images(
    query: str = Query(..., description="Text query"),
    top_k: int = Query(5, ge=1, le=20)
):
    # Encode text query
    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        text_features = text_outputs.pooler_output
    
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_embedding = text_features.cpu().numpy().astype("float32")


    # FAISS search
    scores, indices = faiss_index.search(text_embedding, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "image_url": image_urls[idx],
            "score": float(score)
        })


    return {
        "query": query,
        "top_k": top_k,
        "results": results
    }
