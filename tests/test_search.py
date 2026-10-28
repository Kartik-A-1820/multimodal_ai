import io
import json
import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import settings
from app import create_app
from build_index import build
from search_runtime import SearchRuntime, normalize
class Encoder:
    def text(self, text):
        return np.array([[3., 0.]], dtype=np.float32)
    def image(self, image):
        return normalize([[image.getpixel((0, 0))[0] + 1., 1.]])
def runtime():
    index = faiss.IndexFlatIP(2)
    index.add(normalize([[1, 0], [0, 1]]))
    return SearchRuntime(index, ['one.png', 'two.png'], Encoder())
def test_ranking_and_alignment():
    assert runtime().search([[10, 0]], 100) == ['one.png', 'two.png']
    with pytest.raises(ValueError):
        SearchRuntime(runtime().index, ['wrong.png'], Encoder())
    with pytest.raises(ValueError):
        normalize([[0., 0.]])
def test_api():
    with TestClient(create_app(runtime)) as client:
        assert client.get('/health').json()['images'] == 2
        assert client.get('/search_by_text', params={'query': 'red'}).json()['results'][0] == 'one.png'
        assert client.get('/search_by_text', params={'query': 'x', 'top_k': -1}).status_code == 422
        assert client.get('/search_by_text', params={'query': ' '}).status_code == 422
        assert client.post('/search_by_image', files={'file': ('bad.png', b'bad')}).status_code == 400
        output = io.BytesIO()
        Image.new('RGB', (2, 2), 'red').save(output, 'PNG')
        assert client.post('/search_by_image', files={'file': ('ok.png', output.getvalue())}).status_code == 200
        assert client.get('/images/not-indexed.png').status_code == 404
def test_index_row_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'IMAGE_DIR', tmp_path)
    for attr, name in [('CSV_PATH', 'captions.csv'), ('INDEX_PATH', 'search.index'), ('METADATA_PATH', 'search.json')]:
        monkeypatch.setattr(settings, attr, tmp_path / name)
    Image.new('RGB', (2, 2), 'red').save(tmp_path / 'one.png')
    settings.CSV_PATH.write_text('image,caption\nmissing.png,absent\none.png,red\none.png,duplicate\n')
    assert build(Encoder()) == 1
    assert json.loads(settings.METADATA_PATH.read_text())['images'] == ['one.png']
    assert faiss.read_index(str(settings.INDEX_PATH)).ntotal == 1
    with pytest.raises(ValueError):
        settings.image_path('../outside.png')

def test_ui_fetches_private_backend_images_on_server(monkeypatch):
    import ui_client
    calls = []
    class Response:
        content = b'actual image bytes'
        def raise_for_status(self):
            calls.append('checked')
    def get(url, timeout):
        calls.append((url, timeout))
        return Response()
    monkeypatch.setattr(ui_client.requests, 'get', get)
    assert ui_client.fetch_result_image('http://127.0.0.1:8000/', 'folder/my image.png') == b'actual image bytes'
    assert calls == [('http://127.0.0.1:8000/images/folder%2Fmy%20image.png', 60), 'checked']
