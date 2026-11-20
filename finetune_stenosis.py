"""
Stenosis Crop-Based Binary Classifier
=======================================
Model   : DenseNet121 (RadImageNet pretrained)
Dataset : ARCADE stenosis - bounding box crops
Task    : BINARY classification
          Class 0 - Normal vessel region (random crops from images)
          Class 1 - Stenosis (crops from annotated bounding boxes)

Why crop-based:
  Stenosis regions are ~40x50px in a 512x512 image.
  A full-image classifier cannot learn to count tiny objects.
  Crop-based approach gives the model the actual stenosis texture.
"""

import argparse, json, os, shutil, random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import mlflow
from PIL import Image
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, f1_score)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',   default='arcade/stenosis')
parser.add_argument('--ckpt',       default='ckpt/DenseNet121.pt')
parser.add_argument('--save_dir',   default='ckpt')
parser.add_argument('--n_epochs',   type=int,   default=40)
parser.add_argument('--batch_size', type=int,   default=32)
parser.add_argument('--lr',         type=float, default=3e-4)
parser.add_argument('--size',       type=int,   default=96,
                    help='Crop size fed to model (stenosis regions are small)')
parser.add_argument('--device',     default='cuda')
parser.add_argument('--name',       required=True)
parser.add_argument('--exp_name',   default='stenosis_binary')
parser.add_argument('--seed',       type=int,   default=42)
parser.add_argument('--neg_ratio',  type=float, default=1.5,
                    help='Negative crops per positive crop ratio')
opt = parser.parse_args()

random.seed(opt.seed)
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
print(f'Device : {device}')
if device.type == 'cuda':
    print(f'GPU    : {torch.cuda.get_device_name(0)}')
    print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')

CLASS_NAMES = ['Normal', 'Stenosis']

# ── Crop Extractor ─────────────────────────────────────────────────────────────
def iou(boxA, boxB):
    """IoU between two [x,y,w,h] boxes."""
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0]+boxA[2], boxA[1]+boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0]+boxB[2], boxB[1]+boxB[3]
    inter_w = max(0, min(ax2,bx2) - max(ax1,bx1))
    inter_h = max(0, min(ay2,by2) - max(ay1,by1))
    inter   = inter_w * inter_h
    union   = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
    return inter / (union + 1e-6)

def build_crop_samples(json_path, img_dir, neg_ratio=1.5, min_box=20, padding=10):
    """
    Returns list of (img_path, bbox_xywh_or_None, label)
    label 1 = stenosis crop, label 0 = random negative crop
    """
    with open(json_path) as f:
        data = json.load(f)

    # Map image_id → file path and size
    img_meta = {i['id']: i for i in data['images']}

    # Collect stenosis bboxes per image
    from collections import defaultdict
    stenosis_boxes = defaultdict(list)   # img_id -> list of [x,y,w,h]
    for ann in data['annotations']:
        if ann['category_id'] == 26:
            stenosis_boxes[ann['image_id']].append(ann['bbox'])

    positive_samples = []
    negative_samples = []

    for img_id, boxes in stenosis_boxes.items():
        meta     = img_meta[img_id]
        img_path = os.path.join(img_dir, meta['file_name'])
        W, H     = meta['width'], meta['height']

        # Positive: each stenosis bounding box (with padding)
        for box in boxes:
            x, y, w, h = box
            if w < min_box or h < min_box:
                continue
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(W - x, w + 2*padding)
            h = min(H - y, h + 2*padding)
            positive_samples.append((img_path, [x, y, w, h], 1))

        # Negative: random crops not overlapping with any stenosis box
        n_neg = max(1, int(len(boxes) * neg_ratio))
        attempts = 0
        added    = 0
        while added < n_neg and attempts < 50:
            attempts += 1
            rw = random.randint(40, 120)
            rh = random.randint(40, 120)
            rx = random.randint(0, max(0, W - rw))
            ry = random.randint(0, max(0, H - rh))
            rand_box = [rx, ry, rw, rh]
            # Reject if overlaps any stenosis box
            overlap = any(iou(rand_box, sb) > 0.1 for sb in boxes)
            if not overlap:
                negative_samples.append((img_path, rand_box, 0))
                added += 1

    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)
    print(f'  Positive (stenosis) crops : {len(positive_samples)}')
    print(f'  Negative (normal)   crops : {len(negative_samples)}')
    return all_samples


# ── Dataset ────────────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        x, y, w, h = [int(v) for v in bbox]
        crop = img.crop((x, y, x+w, y+h))
        return self.transform(crop), label


def get_transforms(size, mode='train'):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if mode == 'train':
        return T.Compose([
            T.Resize((size, size)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.3),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.4, contrast=0.4),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(ckpt_path, num_classes=2):
    model = models.densenet121(weights=None)

    # Load RadImageNet weights: backbone.0.X → features.X
    rad_state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    new_state  = {}
    for k, v in rad_state.items():
        if k.startswith('backbone.0.'):
            new_state['features.' + k[len('backbone.0.'):]] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f'RadImageNet loaded — missing: {len(missing)}, unexpected: {len(unexpected)}')

    in_features = model.classifier.in_features   # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    print(f'Classifier: {in_features} → 256 → {num_classes}')
    return model


# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            logits = model(imgs.to(device))
            probs  = F.softmax(logits, dim=1)[:,1].cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    acc  = (all_preds == all_labels).mean()
    f1   = f1_score(all_labels, all_preds, average='binary')
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    return acc, auc, f1, all_preds, all_labels


# ── Guard for Windows multiprocessing ─────────────────────────────────────────
if __name__ != '__main__':
    import sys; sys.exit(0)

# ── MLflow ─────────────────────────────────────────────────────────────────────
mlflow.set_experiment(opt.exp_name)
run     = mlflow.start_run(run_name=opt.name)
run_dir = f'mlruns/{run.info.experiment_id}/{run.info.run_id}'
art_dir = f'{run_dir}/artifacts'
os.makedirs(art_dir, exist_ok=True)
mlflow.log_params(vars(opt))

# ── Build Data ─────────────────────────────────────────────────────────────────
print('\nBuilding crop samples from bounding box annotations...')
print('TRAIN:')
train_samples = build_crop_samples(
    f'{opt.dataroot}/train/annotations/train.json',
    f'{opt.dataroot}/train/images',
    neg_ratio=opt.neg_ratio)

print('VAL:')
val_samples = build_crop_samples(
    f'{opt.dataroot}/val/annotations/val.json',
    f'{opt.dataroot}/val/images',
    neg_ratio=opt.neg_ratio)

print(f'\nTotal train: {len(train_samples)} | Total val: {len(val_samples)}')

train_ds = CropDataset(train_samples, get_transforms(opt.size, 'train'))
val_ds   = CropDataset(val_samples,   get_transforms(opt.size, 'val'))

# Simple balanced sampler (1:1 pos:neg)
labels_arr = [s[2] for s in train_samples]
counts     = np.bincount(labels_arr)
weights    = 1.0 / counts[labels_arr]
sampler    = WeightedRandomSampler(torch.FloatTensor(weights), len(weights), True)

train_dl = DataLoader(train_ds, batch_size=opt.batch_size, sampler=sampler,
                      num_workers=0, pin_memory=True)
val_dl   = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False,
                      num_workers=0, pin_memory=True)

# ── Build Model ────────────────────────────────────────────────────────────────
print('\nLoading RadImageNet DenseNet121...')
model = build_model(opt.ckpt, num_classes=2).to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable params: {trainable:,}')

# Binary cross entropy — no class weights needed (sampler balances)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=opt.lr,
    steps_per_epoch=len(train_dl), epochs=opt.n_epochs,
    pct_start=0.1)

# ── Train ──────────────────────────────────────────────────────────────────────
print('\n' + '='*60)
print(f'Training BINARY stenosis detector on {device} (RTX 3060)')
print('='*60)

best_auc = 0.0
step     = 0
best_ckpt = f'{opt.save_dir}/best_stenosis_binary.pt'

for epoch in range(opt.n_epochs):
    model.train()
    losses = []
    pbar = tqdm(train_dl, desc=f'Epoch {epoch+1}/{opt.n_epochs}')
    for imgs, labels in pbar:
        step += 1
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
        if step % 20 == 0:
            mlflow.log_metric('train_loss', loss.item(), step=step)

    avg_loss = np.mean(losses)
    val_acc, val_auc, val_f1, preds, labels_np = evaluate(model, val_dl, device)

    pos_acc = (preds[labels_np==1]==1).mean() if (labels_np==1).sum()>0 else 0
    neg_acc = (preds[labels_np==0]==0).mean() if (labels_np==0).sum()>0 else 0

    print(f'\nEpoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} | '
          f'AUC: {val_auc:.4f} | F1: {val_f1:.4f}')
    print(f'  Stenosis detected: {pos_acc*100:.1f}%  |  Normal correct: {neg_acc*100:.1f}%')

    mlflow.log_metrics({'val_acc': val_acc, 'val_auc': val_auc,
                        'val_f1': val_f1, 'epoch_loss': avg_loss}, step=epoch)

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'val_auc': val_auc, 'input_size': opt.size,
                    'class_names': CLASS_NAMES}, best_ckpt)
        print(f'  ✅ Best model saved! AUC: {val_auc:.4f}')

# ── Final Report ───────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('FINAL EVALUATION')
print('='*60)
best_state = torch.load(best_ckpt, map_location=device, weights_only=False)
model.load_state_dict(best_state['model'])
acc, auc, f1, final_preds, final_labels = evaluate(model, val_dl, device)
print(f'Accuracy : {acc:.4f} ({acc*100:.1f}%)')
print(f'AUC      : {auc:.4f}')
print(f'F1       : {f1:.4f}')
print('\nClassification Report:')
print(classification_report(final_labels, final_preds, target_names=CLASS_NAMES))
print('Confusion Matrix:')
print(confusion_matrix(final_labels, final_preds))

mlflow.log_metrics({'final_acc': acc, 'final_auc': auc, 'final_f1': f1})
mlflow.end_run()
print(f'\nDone! Best model: {best_ckpt}  (AUC: {best_auc:.4f})')
