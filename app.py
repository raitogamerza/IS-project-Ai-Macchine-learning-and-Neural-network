import streamlit as st

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="ML & NN Project",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================
# Custom CSS
# ============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 4rem 3rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(100, 100, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(255, 100, 100, 0.08) 0%, transparent 50%);
        animation: float 20s ease-in-out infinite;
    }

    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(5deg); }
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.8);
        max-width: 700px;
        margin: 0 auto;
        line-height: 1.8;
        position: relative;
        z-index: 1;
    }

    .project-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.4s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        height: 100%;
    }

    .project-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 40px rgba(100, 100, 255, 0.2);
        border-color: rgba(100, 100, 255, 0.3);
    }

    .card-icon { font-size: 3rem; margin-bottom: 1rem; }

    .card-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .card-desc {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .card-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem 0.15rem;
    }

    .info-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    .feature-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Hero Section
# ============================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🤖 Machine Learning & Neural Network</div>
    <div class="hero-subtitle">
        โปรเจคพัฒนาโมเดลด้วย Machine Learning และ Neural Network<br>
        พร้อม EDA, อธิบายทฤษฎี, และทดสอบการทำนายแบบ real-time
    </div>
</div>
""", unsafe_allow_html=True)

# ============================
# Two Project Cards
# ============================
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="project-card">
        <div class="card-icon">📰</div>
        <div class="card-title" style="color: #667eea;">Machine Learning</div>
        <div class="card-title" style="color: #e0e0ff; font-size: 1.1rem;">Fake/Real News Classification</div>
        <div class="card-desc" style="margin: 1rem 0;">
            จำแนกข่าวจริงและข่าวปลอมด้วย <strong style="color: #667eea;">Logistic Regression</strong> 
            ร่วมกับ <strong style="color: #667eea;">TF-IDF Vectorization</strong>
        </div>
        <div>
            <span class="card-badge" style="background: rgba(102,126,234,0.2); color: #667eea;">Logistic Regression</span>
            <span class="card-badge" style="background: rgba(102,126,234,0.2); color: #667eea;">TF-IDF</span>
            <span class="card-badge" style="background: rgba(102,126,234,0.2); color: #667eea;">scikit-learn</span>
            <span class="card-badge" style="background: rgba(102,126,234,0.2); color: #667eea;">NLP</span>
        </div>
        <hr style="border-color: rgba(255,255,255,0.1); margin: 1rem 0;">
        <div class="card-desc">
            📊 <strong>Dataset:</strong> 44,898 ข่าว (Real + Fake)<br>
            📑 <strong>Features:</strong> title, text, subject, date<br>
            🎯 <strong>Task:</strong> Binary Classification
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="project-card">
        <div class="card-icon">🚗</div>
        <div class="card-title" style="color: #f97316;">Neural Network</div>
        <div class="card-title" style="color: #e0e0ff; font-size: 1.1rem;">Car Brand Image Classification</div>
        <div class="card-desc" style="margin: 1rem 0;">
            จำแนกยี่ห้อรถยนต์จากรูปภาพด้วย <strong style="color: #f97316;">Convolutional Neural Network (CNN)</strong> 
            ผ่าน <strong style="color: #f97316;">PyTorch</strong>
        </div>
        <div>
            <span class="card-badge" style="background: rgba(249,115,22,0.2); color: #f97316;">CNN</span>
            <span class="card-badge" style="background: rgba(249,115,22,0.2); color: #f97316;">PyTorch</span>
            <span class="card-badge" style="background: rgba(249,115,22,0.2); color: #f97316;">Computer Vision</span>
            <span class="card-badge" style="background: rgba(249,115,22,0.2); color: #f97316;">Image Classification</span>
        </div>
        <hr style="border-color: rgba(255,255,255,0.1); margin: 1rem 0;">
        <div class="card-desc">
            🖼️ <strong>Dataset:</strong> Cars Dataset (7 brands)<br>
            🚗 <strong>Classes:</strong> Audi, Hyundai Creta, Swift, ...<br>
            🎯 <strong>Task:</strong> Multi-class Classification
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================
# Pages Overview
# ============================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <h3 style="color: #e0e0ff; margin-top: 0;">📋 หน้าเว็บทั้งหมด</h3>
    <table style="width: 100%; color: rgba(255,255,255,0.75); border-collapse: collapse;">
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <td style="padding: 0.75rem 0;">📖 <strong>ML Explanation</strong></td>
            <td style="padding: 0.75rem 0;">EDA, ทฤษฎี TF-IDF & Logistic Regression, ผลลัพธ์โมเดล</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <td style="padding: 0.75rem 0;">🔍 <strong>ML Demo</strong></td>
            <td style="padding: 0.75rem 0;">ทดสอบจำแนกข่าวจริง/ข่าวปลอม แบบ real-time</td>
        </tr>
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <td style="padding: 0.75rem 0;">📖 <strong>NN Explanation</strong></td>
            <td style="padding: 0.75rem 0;">EDA, ทฤษฎี CNN, ผลลัพธ์โมเดล, Training curves</td>
        </tr>
        <tr>
            <td style="padding: 0.75rem 0;">🔍 <strong>NN Demo</strong></td>
            <td style="padding: 0.75rem 0;">ทดสอบจำแนกยี่ห้อรถยนต์จากรูปภาพ</td>
        </tr>
    </table>
</div>
""", unsafe_allow_html=True)

# ============================
# Tech Stack
# ============================
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
    <h3 style="color: #e0e0ff; margin-top: 0;">🛠️ Technology Stack</h3>
    <div style="color: rgba(255,255,255,0.7); line-height: 2;">
        <span class="card-badge" style="background: rgba(102,126,234,0.2); color: #667eea;">Python</span>
        <span class="card-badge" style="background: rgba(102,126,234,0.2); color: #667eea;">Streamlit</span>
        <span class="card-badge" style="background: rgba(102,126,234,0.2); color: #667eea;">scikit-learn</span>
        <span class="card-badge" style="background: rgba(249,115,22,0.2); color: #f97316;">PyTorch</span>
        <span class="card-badge" style="background: rgba(34,197,94,0.2); color: #22c55e;">Pandas</span>
        <span class="card-badge" style="background: rgba(34,197,94,0.2); color: #22c55e;">NumPy</span>
        <span class="card-badge" style="background: rgba(34,197,94,0.2); color: #22c55e;">Matplotlib</span>
        <span class="card-badge" style="background: rgba(34,197,94,0.2); color: #22c55e;">Seaborn</span>
        <span class="card-badge" style="background: rgba(34,197,94,0.2); color: #22c55e;">WordCloud</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================
# How to Use
# ============================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="info-card">
    <h3 style="color: #e0e0ff; margin-top: 0;">👉 วิธีเริ่มใช้งาน</h3>
    <p style="color: rgba(255,255,255,0.7); line-height: 1.8;">
        เลือกหน้าจาก <strong style="color: #667eea;">sidebar ด้านซ้าย</strong> เพื่อ:<br>
        📖 อ่านคำอธิบายโมเดลและดู EDA<br>
        🔍 ทดสอบการทำนายแบบ real-time
    </p>
</div>
""", unsafe_allow_html=True)

# ============================
# Footer
# ============================
st.markdown("""
<div class="footer">
    Made with ❤️ using Streamlit &nbsp;|&nbsp; Machine Learning & Neural Network Project
</div>
""", unsafe_allow_html=True)
