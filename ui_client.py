"""Backend image requests run in Streamlit's server, not the user's browser."""
from urllib.parse import quote
import requests


def fetch_result_image(api_url, name):
    response = requests.get(
        f"{api_url.rstrip('/')}/images/{quote(name, safe='')}", timeout=60
    )
    response.raise_for_status()
    return response.content
