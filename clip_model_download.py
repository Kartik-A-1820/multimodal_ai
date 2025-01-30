import open_clip
import faiss

# Check available models
available_models = open_clip.list_pretrained()
print("Available models:", available_models)

# Load CLIP model with correct name
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Initialize FAISS index
index = faiss.IndexFlatL2(512)  # 512-D embedding space
print("CLIP Model & FAISS Index initialized successfully!")
