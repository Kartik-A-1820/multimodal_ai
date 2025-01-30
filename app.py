import faiss
import open_clip
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import pandas as pd
import os

# Load FAISS Index
faiss_index_path = "G:/multimodal_ai/models/faiss_index.idx"
index = faiss.read_index(faiss_index_path)

# Load CLIP Model
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Load dataset metadata for retrieving filenames
csv_path = "G:/multimodal_ai/datasets/final_captions.csv"
df = pd.read_csv(csv_path)
image_folder = "G:/multimodal_ai/datasets/Images/"

# Initialize FastAPI
app = FastAPI()

# Function to extract text embeddings
def extract_text_features(text):
    tokens = tokenizer([text])
    with torch.no_grad():
        features = model.encode_text(tokens)
    return features.cpu().numpy()

# Function to extract image embeddings
def extract_image_features(image: Image.Image):
    image = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model.encode_image(image)
    return features.cpu().numpy()

# Text-to-Image Search Endpoint
@app.get("/search_by_text")
def search_by_text(query: str, top_k: int = 5):
    query_embedding = extract_text_features(query)
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    
    # Search in FAISS
    distances, indices = index.search(query_embedding, top_k)
    results = [df.iloc[i]['image'] for i in indices[0]]
    
    return {"query": query, "results": results}

# Image-to-Image Search Endpoint
@app.post("/search_by_image")
def search_by_image(file: UploadFile = File(...), top_k: int = 5):
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    image_embedding = extract_image_features(image)
    image_embedding = np.array(image_embedding, dtype=np.float32).reshape(1, -1)
    
    # Search in FAISS
    distances, indices = index.search(image_embedding, top_k)
    results = [df.iloc[i]['image'] for i in indices[0]]
    
    return {"results": results}
