import streamlit as st
import json
import re
import cv2
import numpy as np
import easyocr
from rapidfuzz import process, fuzz
from PIL import Image

# ==============================
# تحميل الداتا
# ==============================
@st.cache_data
def load_drugs():
    with open("drug_names.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data[0], dict):
        data = [d["name"] for d in data]
    return data

DRUG_DB = load_drugs()

# ==============================
# OCR INIT
# ==============================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ==============================
# IMAGE PROCESSING
# ==============================
def preprocess(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    return thresh

# ==============================
# OCR
# ==============================
def extract_text(img):
    processed = preprocess(img)
    results = reader.readtext(processed)
    text = " ".join([r[1] for r in results])
    return text

# ==============================
# MATCHING
# ==============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def match_drug(text):
    clean = clean_text(text)

    match, score, _ = process.extractOne(
        clean,
        DRUG_DB,
        scorer=fuzz.token_sort_ratio
    )

    if score < 70:
        return None, score

    return match, score

# ==============================
# UI
# ==============================
st.set_page_config(page_title="💊 Drug OCR", layout="centered")

st.title("💊 Drug Recognition System")
st.write("ارفع صورة علبة الدواء وسيتم التعرف عليها")

uploaded_file = st.file_uploader("📷 Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze"):
        with st.spinner("Processing..."):
            text = extract_text(image)
            drug, score = match_drug(text)

        st.subheader("📄 OCR Text")
        st.write(text)

        st.subheader("💊 Prediction")
        if drug:
            st.success(f"{drug}")
            st.write(f"Confidence: {score:.2f}")
        else:
            st.error("❌ لم يتم التعرف على الدواء")