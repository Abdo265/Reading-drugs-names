import streamlit as st
import cv2
import numpy as np
import easyocr
import json
import re
import os
from PIL import Image
from rapidfuzz import process, fuzz
import tempfile

# ──────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Identifier",
    page_icon="💊",
    layout="centered",
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fc; }
    .result-box {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        border-radius: 16px;
        padding: 24px 28px;
        color: white;
        margin-top: 20px;
    }
    .result-box h2 { margin: 0 0 6px 0; font-size: 1.1rem; opacity: .8; }
    .result-box h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .score-badge {
        display: inline-block;
        background: rgba(255,255,255,.2);
        border-radius: 24px;
        padding: 4px 14px;
        font-size: .9rem;
        margin-top: 10px;
    }
    .ocr-box {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 14px 18px;
        font-family: monospace;
        color: #444;
        margin-top: 14px;
    }
    .no-match {
        background: #fff3cd;
        border-radius: 12px;
        padding: 18px 22px;
        color: #856404;
        margin-top: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  Cached resources
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading OCR engine …")
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_data(show_spinner=False)
def load_drug_db(path: str):
    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)
    if db and isinstance(db[0], dict):
        db = [d["name"] for d in db]
    return db

# ──────────────────────────────────────────────
#  Core functions
# ──────────────────────────────────────────────
def preprocess(img):
    img = img.convert("RGB")
    img = np.array(img)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 🔥 أهم خطوة: زيادة التباين
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # تنعيم خفيف
    blur = cv2.GaussianBlur(enhanced, (3,3), 0)

    return blur

def extract_text(img):
    processed = preprocess(img)

    results = reader.readtext(processed)

    # خد أعلى كلمات ثقة
    texts = [r[1] for r in results if r[2] > 0.4]

    # 🔥 خد أول 3 كلمات بس (غالبًا اسم الدوا)
    return " ".join(texts[:3])

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def match_drug(ocr_text: str, drug_db: list, threshold: int = 60):
    clean = clean_text(ocr_text)
    if not clean:
        return None, 0

    match, score, _ = process.extractOne(
        clean, drug_db, scorer=fuzz.partial_ratio
    )
    if score < threshold:
        return None, score
    return match, score

# ──────────────────────────────────────────────
#  Sidebar – settings
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    json_file = st.file_uploader(
        "Upload drug_names.json",
        type=["json"],
        help="JSON list of drug names (strings or dicts with 'name' key)"
    )

    threshold = st.slider(
        "Match confidence threshold", 0, 100, 60,
        help="Results below this score are shown as 'No match'"
    )

    st.markdown("---")
    st.caption("Made with EasyOCR + RapidFuzz")

# ──────────────────────────────────────────────
#  Load DB
# ──────────────────────────────────────────────
DRUG_DB = []

if json_file:
    # save temp file so we can cache it
    tmp_json = os.path.join(tempfile.gettempdir(), "drug_names_upload.json")
    with open(tmp_json, "wb") as f:
        f.write(json_file.read())
    DRUG_DB = load_drug_db(tmp_json)
    st.sidebar.success(f"✅ {len(DRUG_DB)} drugs loaded")
elif os.path.exists("drug_names.json"):
    DRUG_DB = load_drug_db("drug_names.json")
    st.sidebar.info(f"📂 Using local drug_names.json  ({len(DRUG_DB)} drugs)")
else:
    st.sidebar.warning("⚠️ No drug database found. Upload drug_names.json")

# ──────────────────────────────────────────────
#  Main UI
# ──────────────────────────────────────────────
st.title("💊 Drug Identifier")
st.write("Upload a photo of a medication package and we'll identify the drug from your database.")

uploaded = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    # Show image
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Uploaded image", use_container_width=True)

    if not DRUG_DB:
        st.error("❌ Please upload a drug_names.json file in the sidebar first.")
        st.stop()

    with st.spinner("🔍 Running OCR …"):
        reader = load_reader()
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        ocr_text = extract_text(reader, img_bgr)

    with st.spinner("🧠 Matching against drug database …"):
        drug, score = match_drug(ocr_text, DRUG_DB, threshold)

    # ── Result ──
    if drug:
        st.markdown(f"""
        <div class="result-box">
            <h2>Identified Drug</h2>
            <h1>{drug}</h1>
            <span class="score-badge">Confidence: {score}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="no-match">
            <b>⚠️ No confident match found</b><br>
            Best score was <b>{score}%</b> (threshold: {threshold}%).
            Try lowering the threshold or check image quality.
        </div>
        """, unsafe_allow_html=True)

    # ── OCR debug ──
    with st.expander("📝 Raw OCR text"):
        st.markdown(f'<div class="ocr-box">{ocr_text if ocr_text else "(no text detected)"}</div>',
                    unsafe_allow_html=True)

    # ── Top candidates ──
    with st.expander("🔍 Top 5 candidates"):
        if DRUG_DB and ocr_text:
            clean = clean_text(ocr_text)
            top5 = process.extract(clean, DRUG_DB, scorer=fuzz.token_sort_ratio, limit=5)
            for name, sc, _ in top5:
                bar_color = "#1a73e8" if sc >= threshold else "#9e9e9e"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
                  <span style="width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{name}</span>
                  <div style="flex:1;background:#e0e0e0;border-radius:4px;height:10px">
                    <div style="width:{sc}%;background:{bar_color};height:10px;border-radius:4px"></div>
                  </div>
                  <span style="width:40px;text-align:right;font-size:.85rem">{sc}%</span>
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("👆 Upload a drug package image to get started.")