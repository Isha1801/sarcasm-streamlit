import streamlit as st
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np
import os
import requests
import re
import zipfile
import io

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "tf_model.h5")
SHARE_URL = "https://www.dropbox.com/scl/fi/7s8cepv30f5qgtb77itoc/model.zip?rlkey=p2mb6v8jn54ll1l2pa2tle8yu&st=vtvu5yhw&dl=1"

# Download model from Google Drive if not present
def extract_drive_id(url):
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None

def get_direct_download_url(share_url):
    file_id = extract_drive_id(share_url)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(os.path.join(MODEL_DIR, "tf_model.h5")):
        st.info("📥 Downloading zipped model...")
        try:
            r = requests.get(SHARE_URL)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(MODEL_DIR)
            st.success("✅ Model unzipped and ready!")
        except Exception as e:
            st.error(f"❌ Failed to unzip model: {e}")




@st.cache_resource
def load_model():
    try:
        download_model()
        model = TFBertForSequenceClassification.from_pretrained("model", from_pt=False)
        tokenizer = BertTokenizer.from_pretrained("model")
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None



# ========== UI ==========

st.title("🗣️ Sarcasm Detection App")

user_input = st.text_area("Enter a sentence you want to check for sarcasm:")

if st.button("Predict"):
    if user_input.strip():
        tokenizer, model = load_model()
        if model is None:
            st.error("🚫 Model failed to load.")
        else:
            inputs = tokenizer(user_input, padding='max_length', truncation=True, max_length=64, return_tensors="tf")
            outputs = model(inputs, training=False).logits
            prob = tf.nn.softmax(outputs, axis=1)
            prediction = np.argmax(prob)

            result = "Sarcastic 😏" if prediction == 1 else "Not Sarcastic 🙂"
            st.success(f"Prediction: **{result}**")
    else:
        st.warning("⚠️ Please enter a sentence to predict.")
