import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="NN Explanation — Car Classification",
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
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 2rem 0 1.5rem 0;
        box-shadow: 0 4px 20px rgba(249, 115, 22, 0.3);
    }

    .section-header h2 {
        color: white;
        margin: 0;
        font-weight: 600;
    }

    .section-header p {
        color: rgba(255,255,255,0.8);
    }

    .theory-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        color: rgba(255,255,255,0.9);
    }

    .theory-card h3 {
        color: #f97316;
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
        background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
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


    .ref-card a { color: #f97316; text-decoration: none; }
    .ref-card a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ============================
# Page Title
# ============================
st.markdown("""

<div style="text-align: center; padding: 1rem 0;">
<h1>📖 Neural Network — Car Brand Classification</h1>
<p style="font-size: 1.1rem; opacity: 0.8;">
Convolutional Neural Network (ResNet18 Transfer Learning) with PyTorch
</p>
</div>
""", unsafe_allow_html=True)

# ============================
# Load Data
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_DIR = os.path.join(BASE_DIR, "Neural-network", "Dataset", "Cars Dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_data
def load_eda_nn():
    path = os.path.join(MODELS_DIR, "eda_nn_data.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_nn_metrics():

    path = os.path.join(MODELS_DIR, "nn_metrics.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def count_images_per_class(directory):
    counts = {}
    if os.path.exists(directory):
        for cls in sorted(os.listdir(directory)):
            cls_path = os.path.join(directory, cls)
            if os.path.isdir(cls_path):
                counts[cls] = len([f for f in os.listdir(cls_path)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    return counts


eda_nn = load_eda_nn()
nn_metrics = load_nn_metrics()

# ============================
# SECTION 1: EDA
# ============================
st.markdown("""
<div class="section-header">
    <h2>📊 1. Exploratory Data Analysis (EDA)</h2>
    <p>สำรวจและวิเคราะห์ข้อมูลรูปภาพรถยนต์ Cars Dataset</p>
</div>
""", unsafe_allow_html=True)

# --- 1.1 Dataset Overview ---
st.subheader("📋 1.1 Dataset Overview")

train_counts = count_images_per_class(TRAIN_DIR)
test_counts = count_images_per_class(TEST_DIR)
total_train = sum(train_counts.values())
total_test = sum(test_counts.values())
num_classes = len(train_counts)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_train + total_test:,}</div>
        <div class="metric-label">Total Images</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_train:,}</div>
        <div class="metric-label">🏋️ Train Images</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_test:,}</div>
        <div class="metric-label">🧪 Test Images</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{num_classes}</div>
        <div class="metric-label">🚗 Car Brands</div>
    </div>
    """, unsafe_allow_html=True)

# Class names
st.markdown("<br>", unsafe_allow_html=True)
st.write("**ประเภทรถยนต์ทั้งหมด:**")
class_names = sorted(train_counts.keys())
cols = st.columns(len(class_names))
for i, cls in enumerate(class_names):
    with cols[i]:
        st.markdown(f"""
        <div class="metric-card" style="padding: 1rem;">
            <div style="font-size: 1.1rem; font-weight: 600; color: #f97316;">{cls}</div>
            <div class="metric-label">Train: {train_counts.get(cls, 0)} | Test: {test_counts.get(cls, 0)}</div>
        </div>
        """, unsafe_allow_html=True)

# --- 1.2 Class Distribution ---
st.subheader("📊 1.2 การกระจายจำนวนภาพแต่ละประเภท")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0e1117')

colors = ['#f97316', '#fb923c', '#fdba74', '#fed7aa', '#ffedd5', '#fff7ed', '#fbbf24']

# Train distribution
axes[0].set_facecolor('#0e1117')
bars = axes[0].bar(list(train_counts.keys()), list(train_counts.values()),
                   color=colors[:len(train_counts)], edgecolor='white', linewidth=0.5)
axes[0].set_title("Train Set Distribution", color="white", fontsize=13, pad=15)
axes[0].set_ylabel("Number of Images", color="white")
axes[0].tick_params(colors="white", rotation=45)
for bar, val in zip(bars, train_counts.values()):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(val), ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)

# Test distribution
axes[1].set_facecolor('#0e1117')
bars = axes[1].bar(list(test_counts.keys()), list(test_counts.values()),
                   color=colors[:len(test_counts)], edgecolor='white', linewidth=0.5)
axes[1].set_title("Test Set Distribution", color="white", fontsize=13, pad=15)
axes[1].set_ylabel("Number of Images", color="white")
axes[1].tick_params(colors="white", rotation=45)
for bar, val in zip(bars, test_counts.values()):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(val), ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Pie chart
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('#0e1117')
ax.pie(train_counts.values(), labels=train_counts.keys(), colors=colors[:len(train_counts)],
       autopct='%1.1f%%', textprops={'color': 'white', 'fontsize': 11},
       wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}, startangle=90)
ax.set_title("Train Set — Class Proportion", color="white", fontsize=14, pad=20)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# --- 1.3 Sample Images ---
st.subheader("🖼️ 1.3 ตัวอย่างรูปภาพจากแต่ละประเภท")

for cls_name in class_names:
    st.write(f"**🚗 {cls_name}:**")
    cls_dir = os.path.join(TRAIN_DIR, cls_name)
    if os.path.exists(cls_dir):
        images = [f for f in os.listdir(cls_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))][:5]
        if images:
            cols = st.columns(5)
            for i, img_name in enumerate(images):
                with cols[i]:
                    try:
                        img = Image.open(os.path.join(cls_dir, img_name))
                        st.image(img, caption=img_name, use_container_width=True)
                    except Exception:
                        st.write("(ไม่สามารถโหลดรูปได้)")

# --- 1.4 Image Size Analysis ---
st.subheader("📐 1.4 วิเคราะห์ขนาดรูปภาพ")


@st.cache_data
def analyze_image_sizes(directory, sample_size=200):
    widths, heights = [], []
    for cls_name in sorted(os.listdir(directory)):
        cls_dir = os.path.join(directory, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        images = [f for f in os.listdir(cls_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        for img_name in images[:sample_size // len(os.listdir(directory))]:
            try:
                img = Image.open(os.path.join(cls_dir, img_name))
                w, h = img.size
                widths.append(w)
                heights.append(h)
            except Exception:
                pass
    return widths, heights


widths, heights = analyze_image_sizes(TRAIN_DIR)

if widths:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')

    axes[0].set_facecolor('#0e1117')
    axes[0].hist(widths, bins=30, color='#f97316', alpha=0.7, edgecolor='white', linewidth=0.3)
    axes[0].set_title("Distribution of Image Widths", color="white", fontsize=13)
    axes[0].set_xlabel("Width (px)", color="white")
    axes[0].set_ylabel("Frequency", color="white")
    axes[0].tick_params(colors="white")

    axes[1].set_facecolor('#0e1117')
    axes[1].hist(heights, bins=30, color='#fb923c', alpha=0.7, edgecolor='white', linewidth=0.3)
    axes[1].set_title("Distribution of Image Heights", color="white", fontsize=13)
    axes[1].set_xlabel("Height (px)", color="white")
    axes[1].set_ylabel("Frequency", color="white")
    axes[1].tick_params(colors="white")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Width", f"{np.mean(widths):.0f} px")
    with col2:
        st.metric("Avg Height", f"{np.mean(heights):.0f} px")
    with col3:
        st.metric("Min Size", f"{min(widths)}×{min(heights)}")
    with col4:
        st.metric("Max Size", f"{max(widths)}×{max(heights)}")

# ============================
# SECTION 2: Data Preparation
# ============================
st.markdown("""
<div class="section-header">
    <h2>🔧 2. การเตรียมข้อมูล (Data Preparation)</h2>
    <p>ขั้นตอนการเตรียมรูปภาพก่อนนำเข้าโมเดล</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="theory-card">
    <h3>📁 2.1 โครงสร้างข้อมูล</h3>
    <p>ข้อมูลจัดเก็บแบบ <strong>ImageFolder</strong> structure — แยกโฟลเดอร์ตามประเภทรถยนต์:</p>
    <pre style="color: rgba(255,255,255,0.7);">
Cars Dataset/
├── train/
│   ├── Audi/
│   ├── Hyundai Creta/
│   ├── Mahindra Scorpio/
│   ├── Rolls Royce/
│   ├── Swift/
│   ├── Tata Safari/
│   └── Toyota Innova/
└── test/
    └── (same structure)
    </pre>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="theory-card">
    <h3>🔄 2.2 Image Preprocessing & Data Augmentation</h3>
    <ul>
        <li><strong>Resize</strong>: ปรับขนาดรูปทั้งหมดเป็น 128×128 pixels</li>
        <li><strong>RandomVerticalFlip & RandomHorizontalFlip</strong>: สุ่มพลิกรูปบน-ล่าง ขวา-ซ้าย (Data Augmentation)</li>
        <li><strong>RandomRotation(25°)</strong>: สุ่มหมุนรูป ±25 องศา</li>
        <li><strong>RandomAffine</strong>: แปลงรูปภาพแบบสุ่ม (Translate, Scale, Shear)</li>
        <li><strong>ColorJitter & GaussianBlur</strong>: สุ่มปรับ brightness, contrast และทำรูปเบลอ</li>
        <li><strong>Normalize</strong>: ใช้ค่า mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet standard)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.code("""
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
""", language="python")

# ============================
# SECTION 3: Theory
# ============================
st.markdown("""
<div class="section-header">
    <h2>📚 3. ทฤษฎีอัลกอริทึม (Algorithm Theory)</h2>
    <p>ทฤษฎีเบื้องหลัง Convolutional Neural Network (CNN)</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="theory-card">
    <h3>🧠 3.1 Neural Network คืออะไร?</h3>
    <p><strong>Neural Network (โครงข่ายประสาทเทียม)</strong> คือโมเดลทางคณิตศาสตร์ที่ได้รับแรงบันดาลใจจาก
    โครงสร้างของสมองมนุษย์ ประกอบด้วย neurons (nodes) ที่เชื่อมต่อกันเป็น layers:</p>
    <ul>
        <li><strong>Input Layer</strong>: รับข้อมูลเข้า (เช่น pixel ของรูปภาพ)</li>
        <li><strong>Hidden Layers</strong>: ประมวลผลข้อมูล แต่ละ layer เรียนรู้ features ที่ซับซ้อนขึ้น</li>
        <li><strong>Output Layer</strong>: ให้ผลลัพธ์ (เช่น ประเภทรถยนต์)</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="theory-card">
    <h3>🔍 3.2 Convolutional Neural Network (CNN)</h3>
    <p><strong>CNN</strong> เป็น Neural Network ที่ออกแบบมาเฉพาะสำหรับข้อมูลรูปภาพ มี layer พิเศษ:</p>
    <ul>
        <li><strong>Conv2d (Convolutional Layer)</strong>: ใช้ filter/kernel เลื่อนสแกนรูปภาพ เพื่อจับ features 
        เช่น ขอบ, เส้น, texture — filter ขนาด 3×3 เลื่อนไปทั่วรูปภาพและคำนวณ dot product</li>
        <li><strong>BatchNorm2d (Batch Normalization)</strong>: ทำให้ค่า output ของแต่ละ layer มี mean ≈ 0, std ≈ 1 
        ช่วยให้ train เร็วขึ้นและเสถียรขึ้น</li>
        <li><strong>ReLU (Activation Function)</strong>: f(x) = max(0, x) — ตัด ค่าลบออก ช่วยให้โมเดลเรียนรู้ non-linear patterns</li>
        <li><strong>MaxPool2d (Max Pooling)</strong>: ลดขนาดรูปลงครึ่งหนึ่ง โดยเลือกค่ามากสุดในแต่ละ region 
        ช่วยลด computation และป้องกัน overfitting</li>
        <li><strong>Dropout</strong>: สุ่มปิด neurons บางส่วนระหว่าง training เพื่อป้องกัน overfitting</li>
        <li><strong>Flatten + Dense (Fully Connected)</strong>: แปลง feature map เป็น vector แล้วจำแนกประเภท</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="theory-card">
    <h3>🏗️ 3.3 สถาปัตยกรรม CNN ที่ใช้ (Transfer Learning)</h3>
    <p>โมเดลนี้ใช้เทคนิค <strong>Transfer Learning</strong> โดยนำสถาปัตยกรรม <strong>ResNet18 (Pretrained)</strong> รูปแบบโค้ดที่มีการปรับเปลี่ยน Fully Connected layer เพื่อทำ Classification 7 คลาส:</p>
    <pre style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
Base Model: ResNet18 (weights = ResNet18_Weights.DEFAULT)
1. นำเข้าโมเดลและทำการ <strong>Freeze</strong> ทุก layers ก่อน
2. ทำการ <strong>Unfreeze</strong> เฉพาะ Layer ที่ 4 (Layer สุดท้ายของ Conv layers) เพื่อให้ปรับเข้ากับ dataset
3. ทำการ <strong>เปลี่ยน Output Layer (FC)</strong> ใหม่ เพื่อให้มีจำนวน Output Nodes เท่ากับจำนวนคลาสรถยนต์ที่ต้องการทำนาย (7 Classes)

Classifier: Linear(in_features=512, out_features=7, bias=True)
    </pre>
</div>
""", unsafe_allow_html=True)

# ============================
# SECTION 4: Development Steps
# ============================
st.markdown("""
<div class="section-header">
    <h2>⚙️ 4. ขั้นตอนการพัฒนาโมเดล (Model Development)</h2>
    <p>ขั้นตอนการ train โมเดล CNN</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="theory-card">
    <h3>Step 1: สร้างโมเดลแบบ Transfer Learning (ResNet18)</h3>
    <p>โหลด ResNet18 ปรับแต่ง Classifier, สร้าง Loss function (รองรับ Data Imbalance) และระบุ Optimizer ด้วย Learning rate ที่เหมาะสม</p>
</div>
""", unsafe_allow_html=True)

st.code("""
import torch.nn as nn
import torch.optim as optim
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# เปลี่ยน Output Layer (FC) ให้เป็น 7 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# ใช้ CrossEntropyLoss แบบถ่วงน้ำหนักเพื่อจัดการ Data Imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
""", language="python")

st.markdown("""
<div class="theory-card">
    <h3>Step 2: Training Loop</h3>
    <p>Train โมเดลเป็นจำนวน 30 epochs โดยแต่ละ epoch จะ:</p>
    <ul>
        <li>Forward pass: ส่งรูปเข้าโมเดล คำนวณ loss</li>
        <li>Backward pass: คำนวณ gradient แล้วอัปเดต weights</li>
        <li>Validate: ทดสอบกับ test set เพื่อดู performance</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.code("""
for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
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

if nn_metrics:
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{nn_metrics['accuracy']:.2%}</div>
            <div class="metric-label">Test Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{nn_metrics['epochs']}</div>
            <div class="metric-label">Epochs</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(nn_metrics['y_test']):,}</div>
            <div class="metric-label">Test Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{nn_metrics['total_params']:,}</div>
            <div class="metric-label">Parameters</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Training curves
    st.write("**📈 Training & Validation Curves:**")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')
    epochs_range = range(1, nn_metrics['epochs'] + 1)

    # Loss curve
    axes[0].set_facecolor('#0e1117')
    axes[0].plot(epochs_range, nn_metrics['train_losses'], 'o-', color='#f97316', label='Train Loss', linewidth=2)
    axes[0].plot(epochs_range, nn_metrics['val_losses'], 'o-', color='#22d3ee', label='Val Loss', linewidth=2)
    axes[0].set_title("Loss over Epochs", color="white", fontsize=13)
    axes[0].set_xlabel("Epoch", color="white")
    axes[0].set_ylabel("Loss", color="white")
    axes[0].tick_params(colors="white")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # Accuracy curve
    axes[1].set_facecolor('#0e1117')
    axes[1].plot(epochs_range, nn_metrics['train_accs'], 'o-', color='#f97316', label='Train Acc', linewidth=2)
    axes[1].plot(epochs_range, nn_metrics['val_accs'], 'o-', color='#22d3ee', label='Val Acc', linewidth=2)
    axes[1].set_title("Accuracy over Epochs", color="white", fontsize=13)
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Accuracy", color="white")
    axes[1].tick_params(colors="white")
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Classification Report
    st.write("**📋 Classification Report:**")
    st.code(nn_metrics['classification_report'])

    # Confusion Matrix
    st.write("**📊 Confusion Matrix:**")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.heatmap(
        nn_metrics['confusion_matrix'],
        annot=True, fmt='d', cmap='Oranges',
        xticklabels=nn_metrics['class_names'],
        yticklabels=nn_metrics['class_names'],
        ax=ax, annot_kws={"size": 12, "weight": "bold"},
        linewidths=2, linecolor='#0e1117',
    )
    ax.set_xlabel("Predicted Label", color="white", fontsize=13)
    ax.set_ylabel("True Label", color="white", fontsize=13)
    ax.set_title("Confusion Matrix", color="white", fontsize=15, pad=15)
    ax.tick_params(colors="white", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.warning("⚠️ โมเดลยังไม่ได้ train กรุณารัน `python models/train_nn_model.py` ก่อน")

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
<p style="margin-bottom: 1rem;">
    <a href="https://www.kaggle.com/datasets/kshitij192/cars-image-dataset" target="_blank">
        Cars Dataset — Kaggle (Car Brand Classification)
    </a>
</p>

<p style="color: rgba(255,255,255,0.8); margin-bottom: 0.5rem;">📖 <strong>Algorithms & Frameworks:</strong></p>
<ul style="color: rgba(255,255,255,0.6);">
    <li><a href="https://pytorch.org/docs/stable/index.html" target="_blank">PyTorch Documentation</a></li>
    <li><a href="https://pytorch.org/vision/stable/index.html" target="_blank">TorchVision Documentation</a></li>
    <li><a href="https://en.wikipedia.org/wiki/Convolutional_neural_network" target="_blank">Wikipedia — Convolutional Neural Network</a></li>
    <li><a href="https://cs231n.github.io/convolutional-networks/" target="_blank">Stanford CS231n — CNNs for Visual Recognition</a></li>
</ul>

<p style="color: rgba(255,255,255,0.8); margin-bottom: 0.5rem;">🛠️ <strong>Tools & Libraries:</strong></p>
<ul style="color: rgba(255,255,255,0.6);">
    <li><a href="https://streamlit.io/" target="_blank">Streamlit</a></li>
    <li><a href="https://matplotlib.org/" target="_blank">Matplotlib</a></li>
    <li><a href="https://seaborn.pydata.org/" target="_blank">Seaborn</a></li>
    <li><a href="https://pillow.readthedocs.io/" target="_blank">Pillow (PIL)</a></li>
</ul>
</div>
""", unsafe_allow_html=True)
