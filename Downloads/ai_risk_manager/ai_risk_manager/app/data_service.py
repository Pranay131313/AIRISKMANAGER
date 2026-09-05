"""
data_service.py
----------------
Cached data-loading layer shared across all Streamlit pages, so the model
and scored dataset are computed once per session rather than on every
page load.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml.evaluate import score_full_dataset
from db.database import init_db


@st.cache_resource(show_spinner="Loading model and scoring transactions...")
def get_scored_data():
    init_db()
    fdf, spikes, metrics, model, feature_cols = score_full_dataset()
    return fdf, spikes, metrics, model, feature_cols
