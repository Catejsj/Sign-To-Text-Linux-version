# %% [markdown]
# # 🤟 Khmer Sign Language — Model Training
#
# **Self-contained notebook for Google Colab.**
# No GitHub access needed — all code is in the cells below.
#
# ## How to use:
# 1. Upload your `data/sequences/` folder to Google Drive
# 2. Open this notebook in Colab
# 3. Set GPU runtime: Runtime → Change runtime type → T4 GPU
# 4. Run All (Ctrl+F9)
# 5. Download the trained weights from your Drive

# %% [markdown]
# ## 1. Install Dependencies

# %%
# !pip install torch numpy matplotlib scikit-learn --quiet

# %% [markdown]
# ## 2. Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')

# Set your paths here ↓
DATA_DIR    = '/content/drive/MyDrive/KSL Project/data/sequences'
WEIGHTS_DIR = '/content/drive/MyDrive/KSL Project/models/weights'
MODEL_NAME  = 'ksl_model_v1.pth'

import os
os.makedirs(WEIGHTS_DIR, exist_ok=True)

print(f"Data dir:    {DATA_DIR}")
print(f"Weights dir: {WEIGHTS_DIR}")

# %% [markdown]
# ## 3. Configuration

# %%
# Training hyperparameters
BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 1e-4
TRAIN_RATIO   = 0.8
VAL_RATIO     = 0.1

# Model architecture
SEQ_LEN    = 60
NUM_JOINTS = 51
COORDS     = 3

print("Config loaded ✓")

# %% [markdown]
# ## 4. Model Architecture (Temporal CNN)

# %%
import torch
import torch.nn as nn


class SignLanguageCNN(nn.Module):
    """
    Temporal CNN for sign language classification.
    Input:  (batch, 60, 51, 3)
    Output: (batch, num_classes)
    """

    def __init__(self, num_classes, seq_len=60, num_joints=51, coords=3):
        super().__init__()
        self.seq_len    = seq_len
        self.num_joints = num_joints
        self.coords     = coords
        in_channels     = num_joints * coords  # 153

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        batch = x.size(0)
        x = x.view(batch, self.seq_len, -1)  # (B, 60, 153)
        x = x.permute(0, 2, 1)                # (B, 153, 60)
        x = self.conv_layers(x)                # (B, 256, 60)
        x = x.mean(dim=2)                      # (B, 256)
        x = self.classifier(x)                 # (B, num_classes)
        return x

    def save_weights(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.classifier[-1].out_features,
            'seq_len': self.seq_len,
            'num_joints': self.num_joints,
            'coords': self.coords,
        }, path)

    @classmethod
    def load_from_weights(cls, path, device='cpu'):
        checkpoint  = torch.load(path, map_location=device, weights_only=True)
        num_classes = checkpoint['num_classes']
        model = cls(num_classes,
                    checkpoint.get('seq_len', 60),
                    checkpoint.get('num_joints', 51),
                    checkpoint.get('coords', 3))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model


print("Model architecture defined ✓")

# %% [markdown]
# ## 5. Dataset Loader

# %%
import json
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset


class SignDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir  = Path(data_dir)
        self.sequences = []
        self.labels    = []
        self.label_map = {}
        self.index_map = {}

        label_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])

        for idx, label_dir in enumerate(label_dirs):
            label_name = label_dir.name
            self.label_map[label_name] = idx
            self.index_map[idx] = label_name

            npy_files = sorted(label_dir.glob('*.npy'))
            for f in npy_files:
                self.sequences.append(f)
                self.labels.append(idx)

        # Save label map
        labels_path = self.data_dir / 'labels.json'
        with open(labels_path, 'w') as f:
            json.dump(self.label_map, f, indent=2)

    @property
    def num_classes(self):
        return len(self.label_map)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = np.load(self.sequences[idx]).astype(np.float32)
        label = self.labels[idx]
        return torch.from_numpy(seq), label

    def split(self, train_ratio=0.8, val_ratio=0.1):
        n = len(self)
        indices = list(range(n))
        rng = np.random.RandomState(42)
        rng.shuffle(indices)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return (
            Subset(self, indices[:n_train]),
            Subset(self, indices[n_train:n_train + n_val]),
            Subset(self, indices[n_train + n_val:]),
        )


# Load dataset
dataset = SignDataset(DATA_DIR)
print(f"Dataset: {len(dataset)} sequences, {dataset.num_classes} classes")
print(f"Classes: {list(dataset.label_map.keys())}")

train_set, val_set, test_set = dataset.split(TRAIN_RATIO, VAL_RATIO)
print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

# %% [markdown]
# ## 6. Training Loop

# %%
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

model     = SignLanguageCNN(dataset.num_classes, SEQ_LEN, NUM_JOINTS, COORDS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# Track metrics
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels    = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * sequences.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total   += sequences.size(0)

    # ── Validate ──
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            labels    = labels.to(device)
            outputs   = model(sequences)
            loss      = criterion(outputs, labels)

            val_loss    += loss.item() * sequences.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total   += sequences.size(0)

    # Metrics
    t_loss = train_loss / max(train_total, 1)
    v_loss = val_loss   / max(val_total, 1)
    t_acc  = train_correct / max(train_total, 1) * 100
    v_acc  = val_correct   / max(val_total, 1) * 100

    history['train_loss'].append(t_loss)
    history['val_loss'].append(v_loss)
    history['train_acc'].append(t_acc)
    history['val_acc'].append(v_acc)

    scheduler.step(v_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"Train: {t_loss:.4f} ({t_acc:.1f}%)  "
              f"Val: {v_loss:.4f} ({v_acc:.1f}%)")

    # Save best
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        best_path = os.path.join(WEIGHTS_DIR, MODEL_NAME)
        model.save_weights(best_path)

print(f"\n✅ Best validation accuracy: {best_val_acc:.1f}%")
print(f"   Weights saved to: {best_path}")

# %% [markdown]
# ## 7. Training Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['train_loss'], label='Train')
ax1.plot(history['val_loss'],   label='Val')
ax1.set_title('Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid(True)

ax2.plot(history['train_acc'], label='Train')
ax2.plot(history['val_acc'],   label='Val')
ax2.set_title('Accuracy (%)')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Evaluation — Confusion Matrix

# %%
from sklearn.metrics import confusion_matrix, classification_report
import itertools

model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        outputs   = model(sequences)
        preds     = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

# Classification report
label_names = [dataset.index_map[i] for i in range(dataset.num_classes)]
print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix')
plt.colorbar(im, ax=ax)
tick_marks = np.arange(len(label_names))
ax.set_xticks(tick_marks)
ax.set_xticklabels(label_names, rotation=45, ha='right')
ax.set_yticks(tick_marks)
ax.set_yticklabels(label_names)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    ax.text(j, i, format(cm[i, j], 'd'),
            ha='center', va='center',
            color='white' if cm[i, j] > thresh else 'black')

ax.set_ylabel('True')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 9. Quick Test — Load Weights & Predict a Sample

# %%
# Load the best model
best_path = os.path.join(WEIGHTS_DIR, MODEL_NAME)
test_model = SignLanguageCNN.load_from_weights(best_path, device=str(device))

# Pick a random test sample
sample_seq, sample_label = test_set[0]
sample_input = sample_seq.unsqueeze(0).to(device)

with torch.no_grad():
    logits = test_model(sample_input)
    probs  = torch.softmax(logits, dim=1)
    conf, pred_idx = probs.max(dim=1)

true_name = dataset.index_map[sample_label]
pred_name = dataset.index_map[pred_idx.item()]

print(f"True label:  {true_name}")
print(f"Predicted:   {pred_name} ({conf.item():.1%} confidence)")
print()
print(f"✅ Download your weights file from:")
print(f"   {best_path}")
print(f"   Place it in: models/weights/{MODEL_NAME}")
