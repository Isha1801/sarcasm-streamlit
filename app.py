import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
import os
import requests
import re

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "tf_model.h5")

SHARE_URL = "https://drive.google.com/file/d/15cLCBxGKXVhkYQQA70Pj5eZC_ozvjsCJ/view?usp=sharing"

# Download from Drive
def extract_drive_id(url):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def get_direct_download_url(share_url):
    file_id = extract_drive_id(share_url)
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model weights...")
        url = get_direct_download_url(SHARE_URL)
        r = requests.get(url)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        st.success("✅ Model downloaded.")

@st.cache_resource
def load_model():
    download_model()
    tokenizer = BertTokenizer.from_pretrained("model")
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.load_weights(MODEL_PATH)
    return tokenizer,
