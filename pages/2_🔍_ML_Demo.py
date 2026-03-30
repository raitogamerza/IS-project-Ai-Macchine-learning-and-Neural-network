import streamlit as st
import os
import re
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="ML Demo — Fake/Real News",
    page_icon="🔍",
    layout="wide",
)

# ============================
# Custom CSS
# ============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .demo-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }

    .demo-header h1 {
        color: white;
        margin: 0 0 0.5rem 0;
    }

    .demo-header p {
        color: rgba(255,255,255,0.7);
        margin: 0;
    }

    .result-card {
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        margin: 1.5rem 0;
    }

    .result-real {
        background: linear-gradient(145deg, #064e3b, #065f46);
        border: 2px solid #22c55e;
    }

    .result-fake {
        background: linear-gradient(145deg, #7f1d1d, #991b1b);
        border: 2px solid #ef4444;
    }

    .result-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }

    .result-label {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }

    .result-confidence {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.8);
    }

    .info-box {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }

    .sample-btn {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Header
# ============================
st.markdown("""
<div class="demo-header">
    <h1>🔍 ML Demo — ทดสอบจำแนกข่าวจริง/ข่าวปลอม</h1>
    <p>พิมพ์หรือวางข่าวภาษาอังกฤษ แล้วกดปุ่ม Predict เพื่อดูผลการจำแนก</p>
</div>
""", unsafe_allow_html=True)

# ============================
# Load Model
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, "ml_model.pkl")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        return model, tfidf
    return None, None


model, tfidf = load_model()

if model is None:
    st.error("⚠️ โมเดลยังไม่ได้ train! กรุณารัน `python models/train_ml_model.py` ก่อน")
    st.stop()


def clean_text(text):
    """Clean text for prediction."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================
# Sample News
# ============================
SAMPLE_REAL = """WASHINGTON (Reuters) - The United States on Thursday imposed sanctions on 
several North Korean officials and entities in response to the country's ballistic missile 
program, the Treasury Department said. The sanctions target officials in North Korea's 
weapons programs and entities that support them, Treasury said in a statement."""

SAMPLE_FAKE = """BREAKING: A secret government document has been leaked showing that the 
president has been conspiring with aliens to take over the world. Anonymous sources confirm 
that the deep state has been hiding alien technology for decades. The mainstream media 
refuses to cover this bombshell story. Share this before they delete it!"""

# ============================
# Input Section
# ============================
st.markdown("""
<div class="info-box">
    <p style="color: rgba(255,255,255,0.8);">
        💡 <strong>วิธีใช้งาน:</strong> พิมพ์หรือวางข้อความข่าวภาษาอังกฤษลงในช่องด้านล่าง 
        แล้วกดปุ่ม <strong>🔍 Predict</strong> หรือเลือกจากตัวอย่างที่เตรียมไว้
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    if st.button("📰 ตัวอย่างข่าวจริง", use_container_width=True):
        st.session_state["news_input"] = SAMPLE_REAL
with col2:
    if st.button("🚫 ตัวอย่างข่าวปลอม", use_container_width=True):
        st.session_state["news_input"] = SAMPLE_FAKE

news_text = st.text_area(
    "📝 ใส่ข่าวที่ต้องการจำแนก:",
    value=st.session_state.get("news_input", ""),
    height=200,
    placeholder="Paste an English news article here...",
)

# ============================
# Prediction
# ============================
if st.button("🔍 Predict", type="primary", use_container_width=True):
    if not news_text.strip():
        st.warning("⚠️ กรุณาใส่ข้อความข่าวก่อนกด Predict")
    else:
        with st.spinner("🔄 กำลังวิเคราะห์..."):
            cleaned = clean_text(news_text)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]

            fake_prob = probabilities[0] * 100
            real_prob = probabilities[1] * 100

            # Result display
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card result-real">
                    <div class="result-icon">✅</div>
                    <div class="result-label">REAL NEWS</div>
                    <div class="result-confidence">ความมั่นใจ: {real_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card result-fake">
                    <div class="result-icon">❌</div>
                    <div class="result-label">FAKE NEWS</div>
                    <div class="result-confidence">ความมั่นใจ: {fake_prob:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Probability chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📊 Probability Distribution")

            fig, ax = plt.subplots(figsize=(8, 3))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')

            categories = ["Fake News", "Real News"]
            probs = [fake_prob, real_prob]
            colors = ["#ef4444", "#22c55e"]

            bars = ax.barh(categories, probs, color=colors, height=0.5, edgecolor="white", linewidth=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", color="white", fontsize=12)
            ax.tick_params(colors="white", labelsize=12)

            for bar, prob in zip(bars, probs):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f'{prob:.1f}%', va='center', color='white', fontweight='bold', fontsize=13)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('rgba(255,255,255,0.3)')
            ax.spines['left'].set_color('rgba(255,255,255,0.3)')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Text analysis details
            with st.expander("🔎 รายละเอียดการวิเคราะห์"):
                st.write(f"**ข้อความดั้งเดิม:** {len(news_text)} ตัวอักษร")
                st.write(f"**ข้อความหลัง clean:** {len(cleaned)} ตัวอักษร")
                st.write(f"**จำนวนคำ:** {len(cleaned.split())}")
                st.write(f"**TF-IDF Features ที่ไม่เป็น 0:** {vectorized.nnz}")

# ============================
# Footer
# ============================
st.markdown("""
---
<div style="text-align: center; padding: 1rem; color: rgba(255,255,255,0.4); font-size: 0.85rem;">
    💡 โมเดลนี้ train จาก Fake and Real News Dataset — ผลการทำนายอาจไม่ถูกต้อง 100% ควรใช้วิจารณญาณในการพิจารณา
</div>
""", unsafe_allow_html=True)
