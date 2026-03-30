"""
Train NN Model — Car Brand Image Classification
CNN with PyTorch
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import Counter
from PIL import Image

# ============================
# 1. Config
# ============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Neural-network", "Dataset", "Cars Dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 30  # ลด epoch ให้เหมาะกับ transfer learning
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ============================
# 2. Data Transforms
# ============================

# เพิ่ม data augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============================
# 3. Load Data
# ============================

print("Loading datasets...")
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

# ใช้ WeightedRandomSampler เพื่อ balance class
from torch.utils.data import WeightedRandomSampler
targets = train_dataset.targets
class_sample_count = np.array([np.sum(np.array(targets) == t) for t in range(len(train_dataset.classes))])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in targets])
samples_weight = torch.from_numpy(samples_weight).double()
sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)

print(f"Classes: {class_names}")
print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

# ============================
# 4. CNN Model
# ============================

# ใช้ ResNet18 pretrained + เปลี่ยน output layer
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# freeze ทุก layerก่อน
for param in model.parameters():
    param.requires_grad = False
# unfreeze เฉพาะ layer4 และ fc
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(DEVICE)
print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================
# 5. Training
# ============================
# Compute class weights to handle data imbalance
from collections import Counter
class_counts = Counter(train_dataset.targets)
total_samples = len(train_dataset)
class_weights = []
for i in range(num_classes):
    weight = total_samples / (num_classes * class_counts[i])
    class_weights.append(weight)
class_weights = torch.FloatTensor(class_weights).to(DEVICE)
print(f"\nClass weights: {dict(zip(class_names, class_weights.tolist()))}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

train_losses = []
train_accs = []
val_losses = []
val_accs = []

print("\nTraining...")
for epoch in range(EPOCHS):
    # Train
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validate
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(test_loader)
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
    scheduler.step(val_loss)

# ============================
# 6. Final Evaluation
# ============================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

accuracy = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names)

print(f"\nFinal Test Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:\n{report}")
print(f"Confusion Matrix:\n{cm}")

# ============================
# 7. Save
# ============================
os.makedirs(MODELS_DIR, exist_ok=True)

# Save model
model_path = os.path.join(MODELS_DIR, "nn_model.pth")
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "num_classes": num_classes,
    "img_size": IMG_SIZE,
}, model_path)

# Save metrics
import joblib

metrics = {
    "accuracy": accuracy,
    "classification_report": report,
    "confusion_matrix": cm,
    "y_test": all_labels,
    "y_pred": all_preds,
    "class_names": class_names,
    "train_losses": train_losses,
    "train_accs": train_accs,
    "val_losses": val_losses,
    "val_accs": val_accs,
    "epochs": EPOCHS,
    "total_params": total_params,
}
joblib.dump(metrics, os.path.join(MODELS_DIR, "nn_metrics.pkl"))

# Save EDA data
train_class_counts = Counter(train_dataset.targets)
test_class_counts = Counter(test_dataset.targets)

eda_nn = {
    "class_names": class_names,
    "num_classes": num_classes,
    "train_total": len(train_dataset),
    "test_total": len(test_dataset),
    "train_class_counts": {class_names[k]: v for k, v in train_class_counts.items()},
    "test_class_counts": {class_names[k]: v for k, v in test_class_counts.items()},
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LEARNING_RATE,
}

# Sample images per class
sample_images = {}
for cls_name in class_names:
    cls_dir = os.path.join(TRAIN_DIR, cls_name)
    images_list = os.listdir(cls_dir)[:3]
    sample_images[cls_name] = [os.path.join(cls_dir, img) for img in images_list]
eda_nn["sample_images"] = sample_images

joblib.dump(eda_nn, os.path.join(MODELS_DIR, "eda_nn_data.pkl"))

print(f"\nSaved to: {MODELS_DIR}")
print("  - nn_model.pth")
print("  - nn_metrics.pkl")
print("  - eda_nn_data.pkl")
print("\nDone!")
