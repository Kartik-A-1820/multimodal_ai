import json
import hashlib
import numpy as np
import settings

def normalize(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Expected finite embedding matrix')
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('Zero embeddings cannot be normalized')
    return np.ascontiguousarray(features / norms)

class ClipEncoder:
    def __init__(self):
        import open_clip
        import torch
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(settings.MODEL, pretrained=settings.PRETRAINED)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(settings.MODEL)
    def text(self, text):
        with self.torch.inference_mode():
            return normalize(self.model.encode_text(self.tokenizer([text])).cpu().numpy())
    def image(self, image):
        with self.torch.inference_mode():
            return normalize(self.model.encode_image(self.preprocess(image).unsqueeze(0)).cpu().numpy())

class SearchRuntime:
    def __init__(self, index, images, encoder):
        if not images or index.ntotal != len(images):
            raise ValueError('Index/metadata row mismatch; rebuild the index')
        self.index, self.images, self.encoder = index, images, encoder
    @classmethod
    def load(cls):
        import faiss
        if not settings.INDEX_PATH.is_file() or not settings.METADATA_PATH.is_file():
            raise FileNotFoundError('Run python build_index.py before starting the API')
        meta = json.loads(settings.METADATA_PATH.read_text())
        if (meta['model'], meta['pretrained']) != (settings.MODEL, settings.PRETRAINED):
            raise ValueError('CLIP settings differ from index; rebuild the index')
        if meta.get('index_sha256') != hashlib.sha256(settings.INDEX_PATH.read_bytes()).hexdigest():
            raise ValueError('Index/metadata hash mismatch; rebuild the index')
        index = faiss.read_index(str(settings.INDEX_PATH))
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError('Legacy L2 index; rebuild for cosine search')
        return cls(index, meta['images'], ClipEncoder())
    def search(self, features, top_k):
        features = normalize(features)
        if features.shape != (1, self.index.d):
            raise ValueError('Query dimension does not match index')
        _, rows = self.index.search(features, min(top_k, len(self.images)))
        return [self.images[int(i)] for i in rows[0] if 0 <= i < len(self.images)]
