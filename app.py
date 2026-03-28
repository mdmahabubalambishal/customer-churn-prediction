# app.py — Hugging Face Spaces entry point
# এটা শুধু app/streamlit_app.py কে call করে

import os
import sys

# Root folder path set করো
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# streamlit_app.py এর content execute করো
exec(open(os.path.join(ROOT_DIR, 'app', 'streamlit_app.py')).read())