# ==============================
#   app.py - نظام التعرف على الأدوية
#   كل حاجة في ملف واحد
# ==============================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import os
import re
import json
import easyocr
from rapidfuzz import process, fuzz
import requests

# ══════════════════════════════
#       إعدادات الصفحة
# ══════════════════════════════
st.set_page_config(
    page_title="💊 نظام التعرف على الأدوية",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════
#   ألوان الثيم (Mazzbot Theme)
# ══════════════════════════════
# Primary Teal  : #2DBD9F
# Dark Teal     : #1A9B82
# Light Mint BG : #EFF9F6
# Dark Text     : #14453A
# Orange Accent : #F5A623
# White Cards   : #FFFFFF
# Border/Divider: #D4EEE8

st.markdown("""
<style>
    /* ── تغيير خلفية الصفحة ── */
    .stApp {
        background-color: #EFF9F6;
    }

    /* ── الشريط الجانبي ── */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #D4EEE8;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #14453A;
    }

    /* ── الهيدر الرئيسي ── */
    .main-header {
        text-align: center;
        padding: 28px 20px;
        background: linear-gradient(135deg, #2DBD9F 0%, #1A9B82 100%);
        border-radius: 18px;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 4px 20px rgba(45, 189, 159, 0.35);
    }
    .main-header h1 {
        font-size: 2.4rem;
        margin: 0 0 8px 0;
        font-weight: 700;
    }
    .main-header p {
        font-size: 1.05rem;
        opacity: 0.92;
        margin: 0;
    }

    /* ── بطاقة النتيجة ── */
    .result-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 24px 28px;
        margin: 15px 0;
        border-left: 5px solid #2DBD9F;
        box-shadow: 0 2px 12px rgba(45, 189, 159, 0.15);
    }
    .result-card h2 {
        color: #14453A;
        margin: 0 0 10px 0;
        font-size: 1.6rem;
    }

    /* ── ثقة النتيجة ── */
    .confidence-high   { color: #1A9B82; font-weight: 700; }
    .confidence-medium { color: #F5A623; font-weight: 700; }
    .confidence-low    { color: #E05252; font-weight: 700; }

    /* ── بطاقات البدائل ── */
    .alt-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 12px 18px;
        margin: 8px 0;
        border: 1px solid #D4EEE8;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: box-shadow 0.2s;
    }
    .alt-card:hover {
        box-shadow: 0 2px 10px rgba(45, 189, 159, 0.2);
    }

    /* ── صندوق نص OCR ── */
    .ocr-text-box {
        background: #14453A;
        color: #7EEBD8;
        border-radius: 12px;
        padding: 16px;
        font-family: monospace;
        font-size: 0.95rem;
        direction: ltr;
        text-align: left;
    }

    /* ── بطاقات الإحصائيات ── */
    .stats-box {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 18px 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(45, 189, 159, 0.12);
        border: 1px solid #D4EEE8;
    }
    .stats-box h3 {
        margin: 0 0 4px 0;
        font-size: 1.7rem;
        color: #14453A;
    }
    .stats-box p {
        margin: 0;
        color: #6B9E94;
        font-size: 0.85rem;
    }

    /* ── أزرار Streamlit ── */
    .stButton > button {
        background: linear-gradient(135deg, #2DBD9F 0%, #1A9B82 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: opacity 0.2s, transform 0.1s;
        box-shadow: 0 3px 10px rgba(45, 189, 159, 0.3);
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #2DBD9F, #1A9B82);
        border-radius: 99px;
    }

    /* ── Dividers ── */
    hr { border-color: #D4EEE8; }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: #FFFFFF;
        border-radius: 10px;
        border: 1px solid #D4EEE8;
        color: #14453A;
        font-weight: 600;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #D4EEE8;
    }
    [data-testid="stMetricValue"] { color: #1A9B82; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════
#       IMAGE PROCESSOR
# ══════════════════════════════
class ImageProcessor:
    @staticmethod
    def preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    @staticmethod
    def preprocess_color(img):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(img, -1, kernel)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def resize(img, max_width=1200):
        h, w = img.shape[:2]
        if w > max_width:
            ratio = max_width / w
            img = cv2.resize(img, (max_width, int(h * ratio)))
        return img


# ══════════════════════════════
#         OCR ENGINE
# ══════════════════════════════
class OCREngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.processor = ImageProcessor()

    def extract_text(self, img_array):
        results = {}

        try:
            enhanced = self.processor.preprocess_color(img_array)
            raw = self.reader.readtext(enhanced)
            text1 = " ".join([r[1] for r in raw])
            conf1 = np.mean([r[2] for r in raw]) if raw else 0
            results['color'] = {'text': text1, 'confidence': conf1, 'details': raw}
        except:
            results['color'] = {'text': '', 'confidence': 0, 'details': []}

        try:
            thresh = self.processor.preprocess(img_array)
            raw2 = self.reader.readtext(thresh)
            text2 = " ".join([r[1] for r in raw2])
            conf2 = np.mean([r[2] for r in raw2]) if raw2 else 0
            results['thresh'] = {'text': text2, 'confidence': conf2, 'details': raw2}
        except:
            results['thresh'] = {'text': '', 'confidence': 0, 'details': []}

        try:
            raw3 = self.reader.readtext(img_array)
            text3 = " ".join([r[1] for r in raw3])
            conf3 = np.mean([r[2] for r in raw3]) if raw3 else 0
            results['original'] = {'text': text3, 'confidence': conf3, 'details': raw3}
        except:
            results['original'] = {'text': '', 'confidence': 0, 'details': []}

        best_method = max(results, key=lambda k: results[k]['confidence'])
        best = results[best_method]

        return {
            'best_text': best['text'],
            'best_confidence': best['confidence'],
            'best_method': best_method,
            'all_results': results,
            'details': best['details']
        }

    def draw_boxes(self, img_array, details):
        img_copy = img_array.copy()
        for (bbox, text, conf) in details:
            pts = np.array(bbox, dtype=np.int32)
            # استخدام ألوان الثيم في المربعات
            if conf > 0.8:
                color = (45, 189, 159)    # تيل (primary)
            elif conf > 0.5:
                color = (245, 166, 35)    # برتقالي (accent)
            else:
                color = (224, 82, 82)     # أحمر (low conf)

            cv2.polylines(img_copy, [pts], True, color, 2)
            x, y = pts[0]
            cv2.putText(
                img_copy, f"{text} ({conf:.0%})",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
        return img_copy


# ══════════════════════════════
#        DRUG MATCHER
# ══════════════════════════════
class DrugMatcher:
    def __init__(self, json_path="drug_names.json"):
        self.drug_list = self._load_db(json_path)
        self.search_db = self._build_search_db()

    def _load_db(self, path):
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f)
        if isinstance(db, list) and len(db) > 0 and isinstance(db[0], dict):
            db = [d.get("name", str(d)) for d in db]
        return db

    def _build_search_db(self):
        search = set()
        for name in self.drug_list:
            search.add(name)
            clean = re.sub(r'["""\']', '', name).strip()
            search.add(clean)
            words = re.findall(r'[A-Za-z]{3,}', name)
            for w in words:
                search.add(w)
            brands = re.findall(r'[""](.*?)[""]', name)
            for b in brands:
                search.add(b)
        return list(search)

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def match(self, ocr_text, top_n=5):
        clean = self.clean_text(ocr_text)

        full_matches = process.extract(
            clean, self.drug_list,
            scorer=fuzz.token_sort_ratio,
            limit=top_n
        )

        words = clean.split()
        word_matches = []
        for word in words:
            if len(word) < 3:
                continue
            m = process.extractOne(word, self.search_db, scorer=fuzz.ratio)
            if m and m[1] > 60:
                full = process.extractOne(m[0], self.drug_list, scorer=fuzz.token_sort_ratio)
                if full:
                    word_matches.append(full)

        if len(words) >= 2:
            two = " ".join(words[:2])
            two_m = process.extractOne(two, self.drug_list, scorer=fuzz.token_sort_ratio)
            if two_m:
                word_matches.append(two_m)

        all_matches = {}
        for name, score, idx in full_matches:
            if name not in all_matches or score > all_matches[name]:
                all_matches[name] = score

        for match_tuple in word_matches:
            name, score, idx = match_tuple
            adjusted = min(score + 5, 100)
            if name not in all_matches or adjusted > all_matches[name]:
                all_matches[name] = adjusted

        sorted_matches = sorted(all_matches.items(), key=lambda x: x[1], reverse=True)[:top_n]

        if not sorted_matches:
            return {'best_match': None, 'confidence': 0, 'alternatives': []}

        best_name, best_score = sorted_matches[0]
        return {
            'best_match': best_name,
            'confidence': best_score,
            'alternatives': [
                {'name': n, 'score': s} for n, s in sorted_matches[1:]
            ]
        }


# ══════════════════════════════
#     تحميل النماذج (Cached)
# ══════════════════════════════
@st.cache_resource
def load_ocr():
    return OCREngine()

@st.cache_resource
def load_matcher():
    json_path = "drug_names.json"
    if not os.path.exists(json_path):
        st.error(f"❌ ملف {json_path} مش موجود!")
        st.stop()
    return DrugMatcher(json_path)


# ══════════════════════════════
#           HEADER
# ══════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>💊 نظام التعرف على الأدوية</h1>
    <p>ارفع صورة علبة الدواء وهنعرفلك اسمه في ثواني</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════
#          SIDEBAR
# ══════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ الإعدادات")

    confidence_threshold = st.slider(
        "🎯 حد الثقة الأدنى", 0, 100, 60, 5,
        help="لو النتيجة أقل من كده هتظهر كـ 'غير متأكد'"
    )
    show_ocr_text = st.checkbox("📝 عرض نص OCR", value=True)
    show_alternatives = st.checkbox("🔄 عرض البدائل", value=True)
    show_boxes = st.checkbox("📦 عرض مربعات النص", value=True)
    num_alternatives = st.slider("عدد البدائل", 1, 10, 5)

    st.divider()
    st.markdown("## 📊 معلومات")
    matcher = load_matcher()

    st.markdown(f"""
    <div class="stats-box" style="margin-top:8px;">
        <h3 style="color:#1A9B82;">{len(matcher.drug_list):,}</h3>
        <p>دواء في القاعدة</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.caption("صنع بـ 💚 بواسطة فريق الصيدلة")


# ══════════════════════════════
#        رفع الصورة
# ══════════════════════════════
st.markdown("### 📷 ارفع صورة الدواء")

upload_col1, upload_col2 = st.columns([2, 1])

with upload_col1:
    uploaded_file = st.file_uploader(
        "اختار صورة علبة الدواء",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        help="ارفع صورة واضحة لعلبة الدواء"
    )

with upload_col2:
    st.markdown("""
    <div style="
        background: #FFFFFF;
        border-radius: 14px;
        padding: 16px 18px;
        border: 1px solid #D4EEE8;
        margin-top: 8px;
    ">
        <p style="color:#14453A; font-weight:600; margin:0 0 8px 0;">💡 نصائح لأفضل نتيجة</p>
        <p style="color:#6B9E94; margin:4px 0;">📸 صورة واضحة ومضيئة</p>
        <p style="color:#6B9E94; margin:4px 0;">🔤 اسم الدواء ظاهر</p>
        <p style="color:#6B9E94; margin:4px 0;">📐 الصورة مش مايلة</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════
#      المعالجة والنتائج
# ══════════════════════════════
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img_array = np.array(image)

    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    # ─── المعالجة ───
    with st.spinner("🔍 جاري تحليل الصورة..."):
        start_time = time.time()

        ocr_engine = load_ocr()
        ocr_result = ocr_engine.extract_text(img_bgr)

        matcher = load_matcher()
        match_result = matcher.match(
            ocr_result['best_text'],
            top_n=num_alternatives + 1
        )

        elapsed = time.time() - start_time

    # ─── عرض الصور ───
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown("#### 📷 الصورة الأصلية")
        st.image(image, use_container_width=True)

    with img_col2:
        if show_boxes and ocr_result['details']:
            st.markdown("#### 📦 النصوص المكتشفة")
            boxed = ocr_engine.draw_boxes(img_bgr, ocr_result['details'])
            st.image(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.markdown("#### 🖼️ الصورة المعالجة")
            processed = ImageProcessor.preprocess_color(img_bgr)
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.divider()

    # ─── إحصائيات ───
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(f"""
        <div class="stats-box">
            <h3>⏱️ {elapsed:.1f}s</h3>
            <p>وقت المعالجة</p>
        </div>""", unsafe_allow_html=True)

    with s2:
        conf = match_result['confidence']
        if conf >= 80:
            cc, ci = "confidence-high", "✅"
        elif conf >= 60:
            cc, ci = "confidence-medium", "⚠️"
        else:
            cc, ci = "confidence-low", "❓"
        st.markdown(f"""
        <div class="stats-box">
            <h3 class="{cc}">{ci} {conf:.0f}%</h3>
            <p>نسبة الثقة</p>
        </div>""", unsafe_allow_html=True)

    with s3:
        nw = len(ocr_result['best_text'].split())
        st.markdown(f"""
        <div class="stats-box">
            <h3>📝 {nw}</h3>
            <p>كلمات مكتشفة</p>
        </div>""", unsafe_allow_html=True)

    with s4:
        st.markdown(f"""
        <div class="stats-box">
            <h3 style="font-size:1.2rem;">🔧 {ocr_result['best_method']}</h3>
            <p>طريقة المعالجة</p>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ─── النتيجة الرئيسية ───
    st.markdown("### 🎯 النتيجة")

    if match_result['best_match'] and match_result['confidence'] >= confidence_threshold:
        drug_name = match_result['best_match']
        confidence = match_result['confidence']

        if confidence >= 80:
            icon, status, color, border = "✅", "تم التعرف بنجاح", "#1A9B82", "#2DBD9F"
        elif confidence >= 60:
            icon, status, color, border = "⚠️", "تم التعرف (ثقة متوسطة)", "#D4860A", "#F5A623"
        else:
            icon, status, color, border = "❓", "غير متأكد", "#B94040", "#E05252"

        st.markdown(f"""
        <div class="result-card" style="border-left-color: {border};">
            <h2>{icon} {drug_name}</h2>
            <p style="font-size: 1.05rem; color: #4A7A70;">
                الحالة: <strong style="color: {color};">{status}</strong>
                &nbsp;|&nbsp;
                الثقة: <strong style="color: {color};">{confidence:.0f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence / 100)

    else:
        st.markdown("""
        <div style="
            background: #FFF5F5;
            border-radius: 14px;
            border: 1px solid #FFCCCC;
            border-left: 5px solid #E05252;
            padding: 20px 24px;
            margin: 10px 0;
        ">
            <p style="color:#B94040; font-size:1.1rem; font-weight:700; margin:0 0 8px 0;">
                ❌ لم يتم التعرف على الدواء
            </p>
            <p style="color:#8B5E5E; margin:0;">
                جرب ارفع صورة أوضح، أو تأكد إن اسم الدواء ظاهر، أو قلل حد الثقة من الإعدادات.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ─── طلب الدواء ───
    st.markdown("### 🛒 طلب الدواء")

    drug_name = match_result.get('best_match', 'Unknown Drug')

    available = st.radio(
        "هل الدواء متوفر عندك؟",
        ["✅ متوفر", "❌ غير متوفر"]
    )

    if available == "❌ غير متوفر":
        address = st.text_area("📍 اكتب عنوانك")
        phone = st.text_input("📞 رقم الهاتف")

        if st.button("🚀 إرسال الطلب للصيدلية"):
            data = {
                "drug_name": drug_name,
                "address": address,
                "phone": phone
            }
            webhook_url = "https://hook.eu1.make.com/eugmkcajoapcapij8lowfd4lo2vbenn7"

            try:
                response = requests.post(webhook_url, json=data)
                if response.status_code == 200:
                    st.success("✅ تم إرسال الطلب للصيدليات")
                else:
                    st.error(f"❌ خطأ: {response.status_code}")
            except Exception as e:
                st.error(f"❌ حصل خطأ: {e}")

    # ─── نص OCR ───
    if show_ocr_text:
        with st.expander("📝 نص OCR المستخرج", expanded=False):
            st.markdown(f"""
            <div class="ocr-text-box">{ocr_result['best_text']}</div>
            """, unsafe_allow_html=True)

            if ocr_result['details']:
                detail_data = [
                    {"الكلمة": text, "الثقة": f"{conf:.0%}"}
                    for (bbox, text, conf) in ocr_result['details']
                ]
                st.table(detail_data)

    # ─── البدائل ───
    if show_alternatives and match_result['alternatives']:
        with st.expander(f"🔄 بدائل محتملة ({len(match_result['alternatives'])})", expanded=False):
            for alt in match_result['alternatives']:
                score = alt['score']
                if score >= 80:
                    badge, clr = "🟢", "#1A9B82"
                elif score >= 60:
                    badge, clr = "🟡", "#D4860A"
                else:
                    badge, clr = "🔴", "#B94040"

                st.markdown(f"""
                <div class="alt-card">
                    <span style="color:#14453A;">{badge} <strong>{alt['name']}</strong></span>
                    <span style="color:{clr}; font-weight:600;">{score:.0f}%</span>
                </div>
                """, unsafe_allow_html=True)

else:
    # ─── حالة عدم رفع صورة ───
    st.markdown("""
    <div style="
        text-align: center;
        padding: 60px 20px;
        background: #FFFFFF;
        border-radius: 18px;
        border: 2px dashed #A8D8CC;
        margin: 30px 0;
    ">
        <div style="font-size: 3.5rem; margin-bottom: 12px;">💊</div>
        <h2 style="color: #14453A; margin: 0 0 8px 0;">ارفع صورة علبة الدواء</h2>
        <p style="color: #6B9E94; margin: 0;">اسحب الصورة هنا أو اضغط Browse files</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════
#           FOOTER
# ══════════════════════════════
st.divider()
st.markdown("""
<div style="text-align: center; color: #6B9E94; padding: 16px;">
    💊 نظام التعرف على الأدوية &nbsp;|&nbsp; صنع بـ 💚
</div>
""", unsafe_allow_html=True)
