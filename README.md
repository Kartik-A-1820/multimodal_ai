# Multimodal AI — CLIP image search

Portable text-to-image and image-to-image retrieval with CLIP, FAISS cosine
similarity, FastAPI, and Streamlit. No paid inference API is required.

## Setup

Python 3.10+:

```bash
git clone https://github.com/Kartik-A-1820/multimodal_ai.git
cd multimodal_ai
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Place legally obtained Flickr images under `datasets/images` (lowercase). The
caption CSV needs an `image` column with filenames relative to that directory;
`datasets/final_captions.csv` is the default. Images are not bundled. The first
CLIP load downloads weights.

```bash
python build_index.py
uvicorn app:app --host 127.0.0.1 --port 8000
# Another activated terminal:
streamlit run ui.py
```

API docs: http://127.0.0.1:8000/docs. UI: http://127.0.0.1:8501.
Stop the API while rebuilding the index, then restart it.

## Configuration

Set environment variables before launching. Relative paths resolve from the
repository, independently of the current working directory.

| Variable | Default |
| --- | --- |
| `MM_IMAGE_DIR` | `datasets/images` |
| `MM_CSV_PATH` | `datasets/final_captions.csv` |
| `MM_INDEX_PATH` | `models/search.index` |
| `MM_METADATA_PATH` | `models/search.json` |
| `MM_CLIP_MODEL` | `ViT-B-32` |
| `MM_CLIP_PRETRAINED` | `openai` |
| `MM_API_URL` (UI only) | `http://127.0.0.1:8000` |

The API and indexer must use identical CLIP settings. Result images are served
by the backend, so the UI needs no local dataset copy.

## Migration and correctness

**Rebuild old indexes.** The supported indexer stores normalized image embeddings
and an exact ordered filename mapping. Missing images are skipped, duplicate CSV
rows are collapsed, and a hash binds metadata to its index. Query and index
vectors use the same normalization and model. Inner product then equals cosine
similarity. The UI/API no longer depend on a Windows drive letter.

The original `feature_extraction.ipynb` and `retrieval_speed_test.ipynb` remain as
historical experiments with machine-specific paths; use `build_index.py` for new
indexes. Their single-query benchmark used mismatched similarity metrics and does
not establish a reproducible speedup. No new semantic-quality or performance
result is claimed by this maintenance update.

## API

- `GET /health`: readiness and image count.
- `GET /search_by_text?query=dog&top_k=5`.
- `POST /search_by_image?top_k=5`: multipart `file`.
- `GET /images/{name}`: indexed images only.

Top-k is restricted to 1–100 and capped to index size. File contents are capped
at 10 MiB and images at 25 million pixels. Use a reverse-proxy body limit before
public exposure; multipart parsing happens before the endpoint's file-size check.
This is a local tool without authentication. Model loading occurs at API startup,
never at import.

## Verification

```bash
pip install -r requirements-test.txt
python -m pytest -q
```

Regression tests exercise real FAISS with deterministic embeddings: filename
alignment after missing/duplicate images, cosine ranking, invalid top-k/blank
queries, malformed uploads, and valid PNG requests. Tests require no model
weights. Real CLIP semantic quality remains a separate labeled-data experiment.

References: [FAISS metrics](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances),
[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/).
