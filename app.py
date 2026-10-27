"""API import never reads indexes or downloads model weights."""
from contextlib import asynccontextmanager
import io
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
import settings
from search_runtime import SearchRuntime

def create_app(runtime_factory=SearchRuntime.load):
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime_factory()
        yield
        app.state.runtime = None
    app = FastAPI(title='Multimodal image search', lifespan=lifespan)
    @app.get('/health')
    def health(request: Request):
        return {'status': 'ready', 'images': len(request.app.state.runtime.images)}
    @app.get('/search_by_text')
    def by_text(request: Request, query: str = Query(min_length=1, max_length=2048), top_k: int = Query(default=5, ge=1, le=100)):
        if not query.strip():
            raise HTTPException(422, 'Query cannot be blank')
        runtime = request.app.state.runtime
        return {'query': query, 'results': runtime.search(runtime.encoder.text(query), top_k)}
    @app.post('/search_by_image')
    def by_image(request: Request, file: UploadFile = File(...), top_k: int = Query(default=5, ge=1, le=100)):
        data = file.file.read(10 * 1024 * 1024 + 1)
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(413, 'Image exceeds 10 MiB')
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > 25_000_000:
                    raise HTTPException(413, 'Image exceeds 25 million pixels')
                features = request.app.state.runtime.encoder.image(image.convert('RGB'))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, 'Invalid image') from exc
        return {'results': request.app.state.runtime.search(features, top_k)}
    @app.get('/images/{name:path}')
    def image_file(name: str, request: Request):
        if name not in request.app.state.runtime.images:
            raise HTTPException(404, 'Image not indexed')
        try:
            path = settings.image_path(name)
        except ValueError as exc:
            raise HTTPException(404, 'Image unavailable') from exc
        if not path.is_file():
            raise HTTPException(404, 'Image unavailable')
        return FileResponse(path)
    return app

app = create_app()
