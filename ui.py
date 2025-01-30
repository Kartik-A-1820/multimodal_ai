import streamlit as st
import requests
from PIL import Image
import io

# FastAPI Endpoints
TEXT_SEARCH_URL = "http://127.0.0.1:8000/search_by_text"
IMAGE_SEARCH_URL = "http://127.0.0.1:8000/search_by_image"

# Streamlit UI Setup
st.title("🔍 Multimodal AI Image Search Engine")
st.sidebar.title("Search Options")
search_type = st.sidebar.radio("Choose Search Type:", ["Text Search", "Image Search"])

# Text-to-Image Search
if search_type == "Text Search":
    query = st.text_input("Enter a text query:")
    if st.button("Search"):
        if query:
            response = requests.get(TEXT_SEARCH_URL, params={"query": query})
            if response.status_code == 200:
                results = response.json()["results"]
                st.subheader("Results:")
                for img in results:
                    st.image(f"G:/multimodal_ai/datasets/Images/{img}", caption=img, width=300)
            else:
                st.error("Failed to fetch results. Check API.")

# Image-to-Image Search
elif search_type == "Image Search":
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
    if uploaded_file and st.button("Search"):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(IMAGE_SEARCH_URL, files=files)
        if response.status_code == 200:
            results = response.json()["results"]
            st.subheader("Results:")
            for img in results:
                st.image(f"G:/multimodal_ai/datasets/Images/{img}", caption=img, width=300)
        else:
            st.error("Failed to fetch results. Check API.")