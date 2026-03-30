import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import matplotlib.font_manager as fm

# ============================
# ตั้งค่า Font สำหรับภาษาไทยใน Matplotlib
# ============================
plt.rcParams['font.family'] = 'Tahoma'

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="ML Explanation — Fake/Real News",
    page_icon="📖",
    layout="wide",
)

# ============================
# Custom CSS
# ============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .section-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 2rem 0 1.5rem 0;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
    }

    .section-header h2 {
        color: white;
        margin: 0;
        font-weight: 600;
    }

    .section-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    .theory-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    .theory-card h3 {
        color: #667eea;
        margin-top: 0;
    }

    .theory-card p, .theory-card li {
        color: rgba(255,255,255,0.75);
        line-height: 1.8;
    }

    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    .ref-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 0.5rem 0;
    }

    .ref-card a {
        color: #667eea;
        text-decoration: none;
    }

    .ref-card a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Page Title
# ============================
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
<h1>📖 Machine Learning — Fake/Real News Classification</h1>
<p style="font-size: 1.1rem; opacity: 0.8;">
Logistic Regression + TF-IDF Vectorization
</p>
</div>
""", unsafe_allow_html=True)

# ============================
# Load Data (Super Auto-Search)
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_data
def load_data():
    true_file = None
    fake_file = None

    # สแกนหาไฟล์ทั่วทั้งโปรเจกต์
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file == "True.csv":
                true_file = os.path.join(root, file)
            elif file == "Fake.csv":
                fake_file = os.path.join(root, file)
        
        # ถ้าเจอทั้งสองไฟล์แล้วก็หยุดค้นหาได้เลย
        if true_file and fake_file:
            break

    if not true_file or not fake_file:
        st.error("❌ หาไฟล์ `True.csv` หรือ `Fake.csv` ไม่เจอเลยครับ กรุณาตรวจสอบว่ามีไฟล์นี้อยู่ในโฟลเดอร์โปรเจกต์ของคุณจริงๆ")
        st.stop()

    # โหลดไฟล์เมื่อเจอ Path ที่ถูกต้อง
    true_df = pd.read_csv(true_file)
    fake_df = pd.read_csv(fake_file)

    true_df["label"] = 1
    fake_df["label"] = 0
    df = pd.concat([true_df, fake_df], ignore_index=True)
    return df, true_df, fake_df


@st.cache_data
def load_metrics():
    metrics_path = os.path.join(MODELS_DIR, "ml_metrics.pkl")
    if os.path.exists(metrics_path):
        return joblib.load(metrics_path)
    return None


@st.cache_data
def load_eda_data():
    eda_path = os.path.join(MODELS_DIR, "eda_data.pkl")
    if os.path.exists(eda_path):
        return joblib.load(eda_path)
    return None

df, true_df, fake_df = load_data()
metrics = load_metrics()
eda_data = load_eda_data()

# ============================
# SECTION 1: EDA
# ============================
st.markdown("""
<div class="section-header">
    <h2>📊 1. Exploratory Data Analysis (EDA)</h2>
    <p>สำรวจและวิเคราะห์โครงสร้างข้อมูล Fake/Real News Dataset</p>
</div>
""", unsafe_allow_html=True)

# --- 1.1 Dataset Overview ---
st.subheader("📋 1.1 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Articles</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(true_df):,}</div>
        <div class="metric-label">✅ Real News</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(fake_df):,}</div>
        <div class="metric-label">❌ Fake News</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df.shape[1]}</div>
        <div class="metric-label">Features</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.write("**ตัวอย่างข้อมูล (Sample Data):**")
st.dataframe(df.head(10), use_container_width=True)

st.write("**ข้อมูลทางสถิติ:**")
st.dataframe(df.describe(), use_container_width=True)

# --- 1.2 Distribution ---
st.subheader("📊 1.2 การกระจายประเภทข่าว (Class Distribution)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0e1117')

# Bar chart
counts = [len(fake_df), len(true_df)]
labels = ["Fake News", "Real News"]
colors = ["#ef4444", "#22c55e"]
bars = axes[0].bar(labels, counts, color=colors, width=0.6, edgecolor="white", linewidth=0.5)
axes[0].set_title("จำนวนข่าวจริง vs ข่าวปลอม", fontsize=14, color="white", pad=15)
axes[0].set_ylabel("จำนวนข่าว", color="white")
axes[0].tick_params(colors="white")
axes[0].set_facecolor('#0e1117')
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 200,
                 f'{count:,}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)

# Pie chart
axes[1].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
            textprops={'color': 'white', 'fontsize': 12},
            wedgeprops={'edgecolor': 'white', 'linewidth': 0.5},
            startangle=90)
axes[1].set_title("สัดส่วนข่าวจริง vs ข่าวปลอม", fontsize=14, color="white", pad=15)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# --- 1.3 Subject Distribution ---
st.subheader("📂 1.3 การกระจายตาม Subject")

col1, col2 = st.columns(2)

with col1:
    st.write("**Real News Subjects:**")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    subject_real = true_df["subject"].value_counts()
    bars = ax.barh(subject_real.index, subject_real.values, color="#22c55e", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Count", color="white")
    ax.tick_params(colors="white")
    ax.set_title("Real News by Subject", color="white", fontsize=13)
    for bar, val in zip(bars, subject_real.values):
        ax.text(val + 50, bar.get_y() + bar.get_height() / 2, f'{val:,}',
                va='center', color='white', fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    st.write("**Fake News Subjects:**")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    subject_fake = fake_df["subject"].value_counts()
    bars = ax.barh(subject_fake.index, subject_fake.values, color="#ef4444", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Count", color="white")
    ax.tick_params(colors="white")
    ax.set_title("Fake News by Subject", color="white", fontsize=13)
    for bar, val in zip(bars, subject_fake.values):
        ax.text(val + 50, bar.get_y() + bar.get_height() / 2, f'{val:,}',
                va='center', color='white', fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# --- 1.4 Text Length Distribution ---
st.subheader("📏 1.4 การกระจายความยาวข้อความ (Text Length Distribution)")

df["text_length"] = df["text"].fillna("").str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0e1117')

# Histogram
axes[0].set_facecolor('#0e1117')
axes[0].hist(df[df["label"] == 1]["text_length"], bins=50, alpha=0.7, label="Real", color="#22c55e")
axes[0].hist(df[df["label"] == 0]["text_length"], bins=50, alpha=0.7, label="Fake", color="#ef4444")
axes[0].set_title("Distribution of Text Length", color="white", fontsize=13)
axes[0].set_xlabel("Text Length (characters)", color="white")
axes[0].set_ylabel("Frequency", color="white")
axes[0].tick_params(colors="white")
axes[0].legend()

# Box plot
axes[1].set_facecolor('#0e1117')
data_to_plot = [
    df[df["label"] == 1]["text_length"].dropna(),
    df[df["label"] == 0]["text_length"].dropna()
]
bp = axes[1].boxplot(data_to_plot, labels=["Real", "Fake"], patch_artist=True)
bp["boxes"][0].set_facecolor("#22c55e")
bp["boxes"][1].set_facecolor("#ef4444")
for element in ['whiskers', 'caps', 'medians']:
    for line in bp[element]:
        line.set_color('white')
axes[1].set_title("Box Plot of Text Length", color="white", fontsize=13)
axes[1].set_ylabel("Text Length (characters)", color="white")
axes[1].tick_params(colors="white")

plt.tight_layout()
st.pyplot(fig)
plt.close()

# --- 1.5 Word Cloud ---
st.subheader("☁️ 1.5 Word Cloud")

@st.cache_resource
def generate_wordcloud(text, color):
    wc = WordCloud(
        width=1000, height=500,
        background_color='#0e1117',
        colormap=color,
        max_words=80,
        relative_scaling=0.5,
        normalize_plurals=False,
        collocations=False,
        contour_width=1,
        contour_color='white'
    ).generate(text)
    return wc

col1, col2 = st.columns(2)

with col1:
    st.write("**☁️ Real News Word Cloud:**")
    real_text = " ".join(true_df["text"].fillna("").head(2000).tolist())
    if real_text.strip():
        wc_real = generate_wordcloud(real_text, "Greens")
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.imshow(wc_real, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

with col2:
    st.write("**☁️ Fake News Word Cloud:**")
    fake_text = " ".join(fake_df["text"].fillna("").head(2000).tolist())
    if fake_text.strip():
        wc_fake = generate_wordcloud(fake_text, "Reds")
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.imshow(wc_fake, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
        plt.close()

# --- 1.6 Top 20 Words ---
st.subheader("🔤 1.6 Top 20 คำที่พบบ่อยที่สุด")

@st.cache_data
def get_top_words(texts, n=20):
    stop_words = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'and', 'is',
                  'it', 'that', 'this', 'was', 'with', 'as', 'are', 'by', 'from', 'or',
                  'be', 'has', 'had', 'have', 'not', 'but', 'he', 'she', 'his', 'her',
                  'they', 'their', 'we', 'you', 'i', 'me', 'my', 'its', 'will', 'would',
                  'said', 'who', 'which', 'been', 'were', 'also', 'than', 'them', 'do',
                  'did', 'about', 'up', 'out', 'if', 'no', 'so', 'what', 'when', 'can',
                  'more', 'there', 'all', 'one', 'two', 's', 'just', 'over', 'new',
                  'after', 'could', 'only', 'into', 'some', 'other', 'time', 'very',
                  'your', 'how', 'may', 'should', 'most', 'any', 'now', 'then', 'such',
                  'reuters', 'u', 't'}
    all_words = []
    for text in texts:
        if isinstance(text, str):
            words = re.findall(r'[a-zA-Z]+', text.lower())
            all_words.extend([w for w in words if w not in stop_words and len(w) > 2])
    return Counter(all_words).most_common(n)


col1, col2 = st.columns(2)

with col1:
    st.write("**Real News — Top 20 Words:**")
    top_real = get_top_words(true_df["text"].head(3000))
    if top_real:
        words, counts_r = zip(*top_real)
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.barh(list(words)[::-1], list(counts_r)[::-1], color="#22c55e", edgecolor="white", linewidth=0.3)
        ax.set_title("Top 20 Words in Real News", color="white", fontsize=13)
        ax.set_xlabel("Frequency", color="white")
        ax.tick_params(colors="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with col2:
    st.write("**Fake News — Top 20 Words:**")
    top_fake = get_top_words(fake_df["text"].head(3000))
    if top_fake:
        words, counts_f = zip(*top_fake)
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.barh(list(words)[::-1], list(counts_f)[::-1], color="#ef4444", edgecolor="white", linewidth=0.3)
        ax.set_title("Top 20 Words in Fake News", color="white", fontsize=13)
        ax.set_xlabel("Frequency", color="white")
        ax.tick_params(colors="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================
# SECTION 2: Data Preparation
# ============================
st.markdown("""
<div class="section-header">
    <h2>🔧 2. การเตรียมข้อมูล (Data Preparation)</h2>
    <p>ขั้นตอนการเตรียมข้อมูลก่อนนำเข้าโมเดล</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="theory-card">
    <h3>📥 2.1 การโหลดและรวมข้อมูล</h3>
    <p>โหลดข้อมูลจากไฟล์ <code>True.csv</code> (ข่าวจริง) และ <code>Fake.csv</code> (ข่าวปลอม) 
    แล้วเพิ่ม label — <strong>1</strong> สำหรับข่าวจริง, <strong>0</strong> สำหรับข่าวปลอม จากนั้นรวมข้อมูลทั้งสองชุดเข้าด้วยกัน</p>
</div>
""", unsafe_allow_html=True)

st.code("""
import pandas as pd

true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

true_df["label"] = 1  # Real News
fake_df["label"] = 0  # Fake News

df = pd.concat([true_df, fake_df], ignore_index=True)
""", language="python")

st.markdown("""
<div class="theory-card">
    <h3>🧹 2.2 การทำความสะอาดข้อความ (Text Cleaning)</h3>
    <ul>
        <li>แปลงข้อความเป็นตัวพิมพ์เล็ก (lowercase)</li>
        <li>ลบ URL, HTML tags</li>
        <li>ลบอักขระพิเศษ เก็บเฉพาะตัวอักษรภาษาอังกฤษ</li>
        <li>ลบช่องว่างซ้ำ</li>
        <li>รวม title + text เป็น feature เดียว</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.code("""
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+|https\\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)                        # Remove HTML
    text = re.sub(r"[^a-zA-Z\\s]", "", text)                 # Keep letters only
    text = re.sub(r"\\s+", " ", text).strip()                # Remove extra spaces
    return text

df["clean_text"] = (df["title"] + " " + df["text"]).apply(clean_text)
""", language="python")

st.markdown("""
<div class="theory-card">
    <h3>🔤 2.3 TF-IDF Vectorization</h3>
    <p>แปลงข้อความเป็นเวกเตอร์ตัวเลขด้วย <strong>TF-IDF (Term Frequency-Inverse Document Frequency)</strong> 
    โดยเลือก top 5,000 features ที่มีความสำคัญสูงสุด</p>
</div>
""", unsafe_allow_html=True)

st.code("""
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
X = tfidf.fit_transform(df["clean_text"])
y = df["label"]
""", language="python")

# ============================
# SECTION 3: Theory
# ============================
st.markdown("""
<div class="section-header">
    <h2>📚 3. ทฤษฎีอัลกอริทึม (Algorithm Theory)</h2>
    <p>ทฤษฎีเบื้องหลัง TF-IDF และ Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="theory-card">
    <h3>📊 3.1 TF-IDF (Term Frequency-Inverse Document Frequency)</h3>
    <p><strong>TF-IDF</strong> เป็นเทคนิคการแปลงข้อความให้อยู่ในรูปเวกเตอร์ตัวเลข 
    โดยวัดความสำคัญของคำแต่ละคำในเอกสารเมื่อเทียบกับทั้ง corpus</p>
    <ul>
        <li><strong>TF (Term Frequency)</strong> = จำนวนครั้งที่คำปรากฏในเอกสาร / จำนวนคำทั้งหมดในเอกสาร</li>
        <li><strong>IDF (Inverse Document Frequency)</strong> = log(จำนวนเอกสารทั้งหมด / จำนวนเอกสารที่มีคำนี้)</li>
        <li><strong>TF-IDF</strong> = TF × IDF</li>
    </ul>
    <p>คำที่ปรากฏบ่อยใน เอกสารหนึ่ง แต่ไม่ค่อยปรากฏในเอกสารอื่น จะได้ค่า TF-IDF สูง 
    → แสดงว่าคำนั้นมีความสำคัญต่อเอกสารนั้น</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="theory-card">
    <h3>🤖 3.2 Logistic Regression</h3>
    <p><strong>Logistic Regression</strong> เป็นอัลกอริทึมสำหรับ Binary Classification 
    ที่ใช้ฟังก์ชัน Sigmoid ในการแปลงค่าผลลัพธ์ให้อยู่ในช่วง 0 ถึง 1 (ความน่าจะเป็น)</p>
    <ul>
        <li><strong>Sigmoid Function</strong>: σ(z) = 1 / (1 + e⁻ᶻ)</li>
        <li><strong>Decision Boundary</strong>: ถ้า σ(z) ≥ 0.5 → ทำนายเป็น class 1 (Real), ถ้า < 0.5 → class 0 (Fake)</li>
        <li><strong>Cost Function</strong>: ใช้ Cross-Entropy Loss เพื่อวัดความผิดพลาดของโมเดล</li>
        <li><strong>Optimization</strong>: ใช้ Gradient Descent เพื่อปรับค่า weights ให้ cost function ต่ำที่สุด</li>
    </ul>
    <p><strong>ข้อดี:</strong> เรียบง่าย, เร็ว, interpret ได้, เหมาะกับ text classification</p>
    <p><strong>ข้อจำกัด:</strong> เป็น linear model, อาจไม่จับ non-linear pattern ได้ดี</p>
</div>
""", unsafe_allow_html=True)

# ============================
# SECTION 4: Development Steps
# ============================
st.markdown("""
<div class="section-header">
    <h2>⚙️ 4. ขั้นตอนการพัฒนาโมเดล (Model Development)</h2>
    <p>ขั้นตอนการ train โมเดลและประเมินผล</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="theory-card">
    <h3>Step 1: Train/Test Split</h3>
    <p>แบ่งข้อมูลเป็น 80% สำหรับ training และ 20% สำหรับ testing โดยใช้ stratified split 
    เพื่อรักษาสัดส่วนของ class ทั้งสองให้เท่ากันในทั้ง train set และ test set</p>
</div>
""", unsafe_allow_html=True)

st.code("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
""", language="python")

st.markdown("""
<div class="theory-card">
    <h3>Step 2: Train Model</h3>
    <p>ฝึก Logistic Regression ด้วย max_iter=1000 เพื่อให้ convergence</p>
</div>
""", unsafe_allow_html=True)

st.code("""
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
""", language="python")

st.markdown("""
<div class="theory-card">
    <h3>Step 3: Evaluate Model</h3>
    <p>ประเมินผลโมเดลด้วย Accuracy, Precision, Recall, F1-Score และ Confusion Matrix</p>
</div>
""", unsafe_allow_html=True)

st.code("""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Fake", "Real"])
cm = confusion_matrix(y_test, y_pred)
""", language="python")

# ============================
# SECTION 5: Results
# ============================
st.markdown("""
<div class="section-header">
    <h2>📈 5. ผลลัพธ์ (Model Results)</h2>
    <p>ผลการประเมินโมเดลบน test set</p>
</div>
""", unsafe_allow_html=True)

if metrics:
    # Accuracy
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{metrics['accuracy']:.2%}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(metrics['y_test']):,}</div>
            <div class="metric-label">Test Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        correct = int((metrics['y_test'] == metrics['y_pred']).sum())
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{correct:,}</div>
            <div class="metric-label">Correct Predictions</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Classification Report
    st.write("**📋 Classification Report:**")
    st.code(metrics['classification_report'])

    # Confusion Matrix
    st.write("**📊 Confusion Matrix:**")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.heatmap(
        metrics['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
        ax=ax,
        annot_kws={"size": 16, "weight": "bold"},
        linewidths=2,
        linecolor='#0e1117',
    )
    ax.set_xlabel("Predicted Label", color="white", fontsize=13)
    ax.set_ylabel("True Label", color="white", fontsize=13)
    ax.set_title("Confusion Matrix", color="white", fontsize=15, pad=15)
    ax.tick_params(colors="white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.warning("⚠️ โมเดลยังไม่ได้ train กรุณารัน `python models/train_ml_model.py` ก่อน")

# ============================
# SECTION 6: References
# ============================
st.markdown("""
<div class="section-header">
    <h2>📚 6. แหล่งอ้างอิง (References)</h2>
    <p>แหล่งข้อมูลและเอกสารอ้างอิง</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="ref-card">
<p style="color: rgba(255,255,255,0.8); margin-bottom: 0.5rem;">📊 <strong>Dataset:</strong></p>
<p style="margin-bottom: 1rem;"><a href="https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset" target="_blank">
Fake and Real News Dataset — Kaggle
</a></p>

<p style="color: rgba(255,255,255,0.8); margin-bottom: 0.5rem;">📖 <strong>Algorithms:</strong></p>
<ul style="color: rgba(255,255,255,0.6);">
    <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html" target="_blank">scikit-learn TfidfVectorizer Documentation</a></li>
    <li><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html" target="_blank">scikit-learn Logistic Regression Documentation</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Tf%E2%80%93idf" target="_blank">Wikipedia — TF-IDF</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Logistic_regression" target="_blank">Wikipedia — Logistic Regression</a></li>
</ul>

<p style="color: rgba(255,255,255,0.8); margin-bottom: 0.5rem;">🛠️ <strong>Tools & Libraries:</strong></p>
<ul style="color: rgba(255,255,255,0.6);">
    <li><a href="https://streamlit.io/" target="_blank">Streamlit</a></li>
    <li><a href="https://pandas.pydata.org/" target="_blank">Pandas</a></li>
    <li><a href="https://matplotlib.org/" target="_blank">Matplotlib</a></li>
    <li><a href="https://seaborn.pydata.org/" target="_blank">Seaborn</a></li>
</ul>
</div>
""", unsafe_allow_html=True)