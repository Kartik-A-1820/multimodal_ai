"""Build normalized image embeddings and the exact FAISS row mapping."""
import csv
import hashlib
import json
import faiss
from PIL import Image
import settings
from search_runtime import ClipEncoder, normalize

def build(encoder=None):
    encoder = encoder or ClipEncoder()
    images, seen, index = [], set(), None
    with settings.CSV_PATH.open(newline='', encoding='utf-8-sig') as handle:
        rows = csv.DictReader(handle)
        if 'image' not in (rows.fieldnames or []):
            raise ValueError('CSV requires an image column')
        for row in rows:
            name = row['image']
            if not name or name in seen:
                continue
            seen.add(name)
            path = settings.image_path(name)
            if not path.is_file():
                print(f'Skipping missing image: {name}')
                continue
            with Image.open(path) as image:
                features = normalize(encoder.image(image.convert('RGB')))
            if index is None:
                index = faiss.IndexFlatIP(features.shape[1])
            index.add(features)
            images.append(name)
    if not images:
        raise ValueError('No images found; check MM_IMAGE_DIR and MM_CSV_PATH')
    settings.INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    settings.METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Rebuild with the API stopped. Hash binding detects partial/mismatched pairs.
    faiss.write_index(index, str(settings.INDEX_PATH))
    settings.METADATA_PATH.write_text(json.dumps({'model': settings.MODEL,
        'pretrained': settings.PRETRAINED, 'images': images,
        'index_sha256': hashlib.sha256(settings.INDEX_PATH.read_bytes()).hexdigest()}, indent=2))
    return len(images)

if __name__ == '__main__':
    print(f'Indexed {build()} images')
