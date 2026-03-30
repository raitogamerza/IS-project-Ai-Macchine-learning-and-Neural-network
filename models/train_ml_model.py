"""
Train ML Model — Fake/Real News Classification
Logistic Regression + TF-IDF
"""

import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================
# 1. Load Data
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Macchine-learning", "Dataset")

print("Loading datasets...")
true_df = pd.read_csv(os.path.join(DATA_DIR, "True.csv", "True.csv"))
fake_df = pd.read_csv(os.path.join(DATA_DIR, "Fake.csv", "Fake.csv"))

# Add labels
true_df["label"] = 1  # Real
fake_df["label"] = 0  # Fake

# Merge
df = pd.concat([true_df, fake_df], ignore_index=True)
print(f"Total samples: {len(df)} (Real: {len(true_df)}, Fake: {len(fake_df)})")

# ============================
# 2. Text Preprocessing
# ============================
def clean_text(text):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Keep only letters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text

print("Cleaning text...")
df["clean_text"] = df["title"].fillna("") + " " + df["text"].fillna("")
df["clean_text"] = df["clean_text"].apply(clean_text)

# ============================
# 3. Train/Test Split (ทำก่อนเพื่อป้องกัน Data Leakage)
# ============================
print("Splitting data...")
X_raw = df["clean_text"]
y = df["label"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train_raw.shape[0]}, Test: {X_test_raw.shape[0]}")

# ============================
# 4. TF-IDF Vectorization
# ============================
print("Vectorizing with TF-IDF (max 5000 features)...")
tfidf = TfidfVectorizer(max_features=5000, stop_words="english")

# ให้โมเดลเรียนรู้คำศัพท์และแปลงข้อมูลเฉพาะบน Train set
X_train = tfidf.fit_transform(X_train_raw)

# แปลงข้อมูล Test set โดยใช้คำศัพท์ที่เรียนรู้มาจาก Train set เท่านั้น
X_test = tfidf.transform(X_test_raw)

# ============================
# 5. Train Model
# ============================
print("Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ============================
# 6. Evaluate
# ============================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Fake", "Real"])
cm = confusion_matrix(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"\nClassification Report:\n{report}")
print(f"Confusion Matrix:\n{cm}")

# ============================
# 7. Save Model & Artifacts
# ============================
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(model, os.path.join(MODELS_DIR, "ml_model.pkl"))
joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))

metrics = {
    "accuracy": accuracy,
    "classification_report": report,
    "confusion_matrix": cm,
    "y_test": np.array(y_test),
    "y_pred": y_pred,
}
joblib.dump(metrics, os.path.join(MODELS_DIR, "ml_metrics.pkl"))

# Save EDA data for the Streamlit app
eda_data = {
    "total_samples": len(df),
    "real_count": int((df["label"] == 1).sum()),
    "fake_count": int((df["label"] == 0).sum()),
    "real_subjects": true_df["subject"].value_counts().to_dict(),
    "fake_subjects": fake_df["subject"].value_counts().to_dict(),
    "text_lengths_real": df[df["label"] == 1]["clean_text"].str.len().tolist(),
    "text_lengths_fake": df[df["label"] == 0]["clean_text"].str.len().tolist(),
    "real_text_sample": " ".join(df[df["label"] == 1]["clean_text"].head(500).tolist()),
    "fake_text_sample": " ".join(df[df["label"] == 0]["clean_text"].head(500).tolist()),
    "word_counts_real": df[df["label"] == 1]["clean_text"].str.split().str.len().tolist(),
    "word_counts_fake": df[df["label"] == 0]["clean_text"].str.split().str.len().tolist(),
}

# Date analysis
df["date"] = pd.to_datetime(df["date"], errors="coerce")
date_counts = df.dropna(subset=["date"]).groupby([df["date"].dt.year, "label"]).size().unstack(fill_value=0)
eda_data["date_analysis"] = date_counts.to_dict()

joblib.dump(eda_data, os.path.join(MODELS_DIR, "eda_data.pkl"))

print(f"\nModels saved to: {MODELS_DIR}")
print("  - ml_model.pkl")
print("  - tfidf_vectorizer.pkl")
print("  - ml_metrics.pkl")
print("  - eda_data.pkl")
print("\nDone!")