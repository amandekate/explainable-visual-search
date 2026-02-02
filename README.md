# Multimodal Image Search

A multimodal image search system that retrieves relevant images using natural language queries or image queries.
The system leverages CLIP embeddings and FAISS vector search to perform efficient semantic retrieval.

# Features

Text → Image Search
Search images using natural language (e.g. “a person”, “city buildings”).

Image → Image Search
Upload an image and find visually similar images.

Fast Semantic Search
Uses FAISS for efficient nearest-neighbor search over image embeddings.

CLIP-based Multimodality
A single embedding space for both text and images.

Offline Embedding Pipeline
Embeddings are generated once and reused for fast inference.

# System Architecture

User Query (Text / Image)

↓

CLIP Encoder (Text / Image)

↓

FAISS Index (Cosine Similarity Search)

↓

Ranked Image Results

# Project Structure

Project Structure
-----------------

.
├── backend/

│ ├── search_api.py # FastAPI inference service

│ ├── run_pipeline.py # Offline embedding pipeline

│ └── data/

│ ├── embeddings/

│ │       └── image_urls.pkl

│ └── faiss_index/

│        └── image_index.faiss

│

├── frontend/ # Next.js UI

│

├── clip_embedding_generation.ipynb

│

└── README.md



# Offline Embedding Pipeline

Image embeddings are generated offline using OpenAI CLIP (ViT-B/32).

Why offline?

Avoids recomputing embeddings on every run

Keeps backend lightweight and fast

Ensures deterministic, reproducible search results

Pipeline steps:

Load image URLs from dataset

Download images temporarily
Generate 512-dim CLIP image embeddings
Normalize embeddings for cosine similarity
Build FAISS IndexFlatIP

Save:

image_index.faiss
image_urls.pkl

These artifacts are committed so the app runs immediately without recomputation.

# Backend (FastAPI)

Run locally:

cd backend
activate virtual environment
uvicorn search_api:app --reload

API Endpoints:

/health
Method: GET
Description: Health check

/search
Method: GET
Description: Text → image search
Example: /search?query=a person

/search-by-image
Method: POST
Description: Image → image search

# Frontend (Next.js)

Run frontend:

cd frontend
npm install
npm run dev

Open in browser:
http://localhost:3000

UI Features:

Text search input

Image upload with preview

Loading and error states

Responsive image grid

Similarity score display

# Technologies Used

Backend:

Python

FastAPI

OpenAI CLIP (ViT-B/32)

FAISS

PyTorch

Frontend:

Next.js (App Router)

React

TypeScript

Tailwind CSS

Tools:

Google Colab (GPU-based embedding generation)

Git and GitHub

# Notes

The dataset file (photos_url.csv) is not committed (input-only).
Virtual environments are excluded via .gitignore.
FAISS index size is kept reasonable for demo purposes.
Cosine similarity is implemented using normalized vectors and inner product.
Key Engineering Decisions

Single CLIP implementation end-to-end
Prevents embedding dimension mismatch.

Metadata alignment
Image URLs are stored in the same order as FAISS vectors.

Stateless backend
All heavy computation is done offline.

Future Improvements

Add image captions for better text relevance

Batch search support

Drag-and-drop image upload

Docker-based deployment

Scalable FAISS indexes (IVF / HNSW)

Author

Aman Dekate
Multimodal systems · Frontend + ML integration · Practical AI projects
