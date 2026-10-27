"""Repository-relative configuration; override using environment variables."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def path(name, default):
    value = Path(os.environ.get(name, default)).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()

IMAGE_DIR = path('MM_IMAGE_DIR', 'datasets/images')
CSV_PATH = path('MM_CSV_PATH', 'datasets/final_captions.csv')
INDEX_PATH = path('MM_INDEX_PATH', 'models/search.index')
METADATA_PATH = path('MM_METADATA_PATH', 'models/search.json')
MODEL = os.environ.get('MM_CLIP_MODEL', 'ViT-B-32')
PRETRAINED = os.environ.get('MM_CLIP_PRETRAINED', 'openai')

def image_path(name):
    candidate = (IMAGE_DIR / name).resolve()
    if not candidate.is_relative_to(IMAGE_DIR):
        raise ValueError('Image path escapes MM_IMAGE_DIR')
    return candidate
