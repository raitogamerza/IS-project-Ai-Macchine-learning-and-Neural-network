import streamlit as st
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="NN Demo — Car Classification",
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

    .demo-header h1 { color: white; margin: 0 0 0.5rem 0; }
    .demo-header p { color: rgba(255,255,255,0.7); margin: 0; }

    .result-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 2px solid #f97316;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(249, 115, 22, 0.2);
        margin: 1.5rem 0;
    }

    .result-icon { font-size: 4rem; margin-bottom: 0.5rem; }

    .result-label {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f97316, #ea580c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
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
</style>
""", unsafe_allow_html=True)

# ============================
# Header
# ============================
st.markdown("""
<div class="demo-header">
    <h1>🔍 NN Demo — ทดสอบจำแนกยี่ห้อรถยนต์</h1>
    <p>อัปโหลดรูปภาพรถยนต์ แล้วกดปุ่ม Predict เพื่อดูผลการจำแนก</p>
</div>
""", unsafe_allow_html=True)

# ============================
IMG_SIZE = 128


# ============================
# Load Model
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_nn_model():
    model_path = os.path.join(MODELS_DIR, "nn_model.pth")
    if not os.path.exists(model_path):
        return None, None, None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    # ใช้ ResNet18 แทน CarCNN
    from torchvision import models
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names, checkpoint.get("img_size", IMG_SIZE)


model, class_names, img_size = load_nn_model()

if model is None:
    st.error("⚠️ โมเดลยังไม่ได้ train! กรุณารัน `python models/train_nn_model.py` ก่อน")
    st.stop()

# Transform for prediction
predict_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================
# Instructions
# ============================
st.markdown(f"""
<div class="info-box">
    <p style="color: rgba(255,255,255,0.8);">
        💡 <strong>วิธีใช้งาน:</strong> อัปโหลดรูปภาพรถยนต์ (.jpg, .jpeg, .png) แล้วกดปุ่ม <strong>🔍 Predict</strong>
    </p>
    <p style="color: rgba(255,255,255,0.6); font-size: 0.9rem;">
        📌 รองรับ {len(class_names)} ประเภท: {', '.join(class_names)}
    </p>
</div>
""", unsafe_allow_html=True)

# ============================
# Upload & Predict
# ============================
uploaded_file = st.file_uploader(
    "📤 อัปโหลดรูปภาพรถยนต์",
    type=["jpg", "jpeg", "png", "webp"],
    help="รองรับไฟล์ .jpg, .jpeg, .png, .webp"
)

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**🖼️ รูปภาพที่อัปโหลด:**")
        st.image(image, use_container_width=True)

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        with st.spinner("🔄 กำลังวิเคราะห์..."):
            # Preprocess
            img_tensor = predict_transform(image).unsqueeze(0)

            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item() * 100
                predicted_class = class_names[predicted_idx]

            # Result
            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-icon">🚗</div>
                    <div class="result-label">{predicted_class}</div>
                    <div class="result-confidence">ความมั่นใจ: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # Probability chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📊 Probability Distribution")

            probs = probabilities.numpy() * 100
            sorted_indices = np.argsort(probs)[::-1]

            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')

            names = [class_names[i] for i in sorted_indices]
            values = [probs[i] for i in sorted_indices]
            colors = ['#f97316' if i == predicted_idx else '#374151' for i in sorted_indices]

            bars = ax.barh(names[::-1], values[::-1],
                           color=colors[::-1], height=0.6,
                           edgecolor='white', linewidth=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", color="white", fontsize=12)
            ax.tick_params(colors="white", labelsize=11)

            for bar, val in zip(bars, values[::-1]):
                if val > 2:
                    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                            f'{val:.1f}%', va='center', color='white',
                            fontweight='bold', fontsize=11)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color((1, 1, 1, 0.3))
            ax.spines['left'].set_color((1, 1, 1, 0.3))

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Top 3 predictions
            st.write("**🏆 Top 3 Predictions:**")
            for rank, idx in enumerate(sorted_indices[:3], 1):
                prob = probs[idx]
                emoji = ["🥇", "🥈", "🥉"][rank - 1]
                st.write(f"{emoji} **{class_names[idx]}** — {prob:.1f}%")

# ============================
# Footer
# ============================
st.markdown("""
---
<div style="text-align: center; padding: 1rem; color: rgba(255,255,255,0.4); font-size: 0.85rem;">
    💡 โมเดลนี้ train จาก Cars Dataset — รองรับ 7 ยี่ห้อ: Audi, Hyundai Creta, Mahindra Scorpio, Rolls Royce, Swift, Tata Safari, Toyota Innova
</div>
""", unsafe_allow_html=True)
