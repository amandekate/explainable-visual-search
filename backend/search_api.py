import pickle
import numpy as np
import clip
import torch
import faiss
from fastapi import FastAPI, Query
from fastapi import UploadFile, File
from PIL import Image
import io

from fastapi.middleware.cors import CORSMiddleware



# Paths

IMAGE_PATHS_PATH = "data/embeddings/image_urls.pkl"
FAISS_INDEX_PATH = "data/faiss_index/image_index.faiss"


# Load FAISS + Metadata

faiss_index = faiss.read_index(FAISS_INDEX_PATH)

print(type(faiss_index)) 

with open(IMAGE_PATHS_PATH, "rb") as f:
    image_paths = pickle.load(f)
    

print("Loaded FAISS index with", faiss_index.ntotal, "vectors")



# Load CLIP (Text Encoder)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()



# FastAPI App

app = FastAPI(title="Multimodal Image Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}




# Search Endpoint

@app.get("/search")
def search_images(
    query: str = Query(..., description="Text query"),
    top_k: int = Query(5, ge=1, le=20)
):
    # Encode text query
    text_tokens = clip.tokenize([query]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    text_embedding = text_features.cpu().numpy().astype("float32")

    scores, indices = faiss_index.search(text_embedding, top_k)

    results = [
        {
            "image_url": image_paths[i],
            "score": float(s)
        }
        for i, s in zip(indices[0], scores[0])
    ]

    return {
        "query": query,
        "top_k": top_k,
        "results": results
    }


@app.post("/search-by-image")
def search_by_image(
    file: UploadFile = File(...),
    top_k: int = 5
):
    # Read uploaded image
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    image_embedding = image_features.cpu().numpy().astype("float32")

    scores, indices = faiss_index.search(image_embedding, top_k)

    results = [
        {
            "image_url": image_paths[i],
            "score": float(s)
        }
        for i, s in zip(indices[0], scores[0])
    ]

    return {
        "top_k": top_k,
        "results": results
    }

