import os
from urllib.parse import quote
import requests
import streamlit as st
API = os.environ.get('MM_API_URL', 'http://127.0.0.1:8000').rstrip('/')
st.title('Multimodal image search')
mode = st.sidebar.radio('Search by', ['Text', 'Image'])
query = st.text_input('Describe an image') if mode == 'Text' else None
upload = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png']) if mode == 'Image' else None
if st.button('Search'):
    try:
        if mode == 'Text':
            if not query or not query.strip():
                st.warning('Enter a description.')
                st.stop()
            response = requests.get(f'{API}/search_by_text', params={'query': query}, timeout=60)
        else:
            if upload is None:
                st.warning('Choose an image.')
                st.stop()
            response = requests.post(f'{API}/search_by_image', files={'file': (upload.name, upload.getvalue(), upload.type)}, timeout=60)
        response.raise_for_status()
        for name in response.json()['results']:
            st.image(f'{API}/images/{quote(name, safe="")}', caption=name, width=300)
    except requests.RequestException as exc:
        st.error(f'Search failed; check backend and MM_API_URL. {exc}')
