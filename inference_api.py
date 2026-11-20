"""
inference_api.py — DeepSA + Stenosis Detection REST Bridge
Called by Node.js backend. Outputs JSON to stdout.
"""
import argparse, json, os, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import cv2
from PIL import Image, ImageChops, ImageOps

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from models import UNet
from datasets import tophat
from utils import fusion_predict, make_mask, clear_mask

parser = argparse.ArgumentParser()
parser.add_argument('--image',     required=True)
parser.add_argument('--out_dir',   required=True)
parser.add_argument('--result_id', required=True)
parser.add_argument('--ckpt',      default=os.path.join(BASE, 'ckpt', 'fscad_36249.ckpt'))
parser.add_argument('--stenosis_ckpt', default=os.path.join(BASE, 'ckpt', 'best_stenosis_binary.pt'))
parser.add_argument('--device',    default='cuda')
opt = parser.parse_args()

device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

# ── Load DeepSA UNet ───────────────────────────────────────────────────────────
def load_deepsa(ckpt_path):
    net = UNet(1, 1, 32, bilinear=True)
    ck  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd  = {k.replace('module.', ''): v for k, v in ck['netE'].items()}
    net.load_state_dict(sd)
    net.eval()
    return net.to(device)

# ── Load Stenosis Detector ─────────────────────────────────────────────────────
def load_stenosis(ckpt_path):
    model = models.densenet121(weights=None)
    in_f  = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4), nn.Linear(in_f, 256),
        nn.ReLU(True), nn.Dropout(0.2), nn.Linear(256, 2)
    )
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    return model.to(device)

# ── Image helpers ──────────────────────────────────────────────────────────────
def save_img(img: Image.Image, name: str) -> str:
    p = os.path.join(opt.out_dir, name)
    img.save(p)
    return f'/results/{opt.result_id}/{name}'

def pct_to_url(url): return url

# ── Run DeepSA segmentation ────────────────────────────────────────────────────
t_start = time.time()

raw_img  = Image.open(opt.image).convert('RGB')
img_gray = raw_img.convert('L')
SIZE     = 512

deepsa_net = load_deepsa(opt.ckpt)

tfmc1 = T.Compose([T.Resize(SIZE), T.Lambda(lambda i: tophat(i, 50)),
                   T.ToTensor(), T.Normalize((0.5,),(0.5,))])
tfmc2 = T.Compose([T.Resize(SIZE), T.ToTensor(), T.Normalize((0.5,),(0.5,))])

x1 = tfmc1(img_gray)
x2 = tfmc2(img_gray)

_, out1 = fusion_predict(deepsa_net, ['none'], x1, multiangle=True,
                         denoise=4, size=SIZE, cutoff=0.4, pad=0, netE=True)
_, out2 = fusion_predict(deepsa_net, ['none'], x2, multiangle=False,
                         denoise=4, size=SIZE, cutoff=0.4, pad=0, netE=True)

arr1 = np.array(out1.convert('L'))
arr2 = np.array(out2.convert('L'))
merged = np.max(np.stack([arr1, arr2]), axis=0)
out_merge = Image.fromarray(merged).convert('RGB')
mask_raw  = make_mask(out_merge, remove_size=2000, local_kernel=21, hole_max_size=100)
out_adj   = T.functional.adjust_gamma(out_merge, 2)
seg_img   = clear_mask(mask_raw)
sub_img   = ImageChops.invert(out_adj)
sub_t     = T.ToTensor()(sub_img) * (2**(-0.5))
sub_img   = T.ToPILImage()(sub_t)

# Resize all to 512×512
orig_512 = raw_img.resize((SIZE, SIZE), Image.BICUBIC)
sub_512  = sub_img.convert('RGB').resize((SIZE, SIZE), Image.BICUBIC)
seg_512  = seg_img.convert('RGB').resize((SIZE, SIZE), Image.BICUBIC)

# Overlay
orig_np  = np.array(orig_512.convert('L'))
orig_rgb = np.stack([orig_np]*3, axis=-1)
mask_np  = np.array(seg_512.convert('L')) > 127
overlay  = orig_rgb.copy()
overlay[mask_np, 0] = np.clip(overlay[mask_np, 0].astype(int) + 80, 0, 255)
overlay[mask_np, 1] = np.clip(overlay[mask_np, 1].astype(int) - 20, 0, 255)
overlay[mask_np, 2] = np.clip(overlay[mask_np, 2].astype(int) + 120, 0, 255)
overlay_img = Image.fromarray(overlay.astype(np.uint8))

# ── Vessel Stats ───────────────────────────────────────────────────────────────
binary = mask_np
vessel_px  = int(binary.sum())
total_px   = binary.size
vessel_pct = vessel_px / total_px * 100

n_comp, lbl, stats_cc, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8))
n_vessels = max(0, n_comp - 1)

# Largest vessel area
if n_vessels > 0:
    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max())
    mean_area = float(areas.mean())
else:
    max_area = 0
    mean_area = 0.0

# ── Stenosis Detection ─────────────────────────────────────────────────────────
stenosis_model = load_stenosis(opt.stenosis_ckpt)

# Slide a window over the sub/vessel image to find suspicious regions
crop_size  = 96
step       = 64
mean_norm  = [0.485, 0.456, 0.406]
std_norm   = [0.229, 0.224, 0.225]
tfm_crop   = T.Compose([T.Resize((crop_size, crop_size)),
                         T.ToTensor(), T.Normalize(mean_norm, std_norm)])

vessel_arr = np.array(sub_512.convert('L'))
h, w       = vessel_arr.shape
crops, positions = [], []

for y in range(0, h - crop_size, step):
    for x in range(0, w - crop_size, step):
        crop = sub_512.crop((x, y, x + crop_size, y + crop_size))
        crops.append(tfm_crop(crop))
        positions.append((x, y))

max_stenosis_prob = 0.0
stenosis_regions  = []

if crops:
    batch = torch.stack(crops).to(device)
    with torch.no_grad():
        logits = stenosis_model(batch)
        probs  = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    max_stenosis_prob = float(probs.max())

    # Collect high-probability regions
    threshold = 0.65
    for prob, (x, y) in zip(probs, positions):
        if prob > threshold:
            stenosis_regions.append({
                'x': x, 'y': y,
                'w': crop_size, 'h': crop_size,
                'prob': float(prob)
            })

# Overall diagnosis
is_stenosis = max_stenosis_prob > 0.60
confidence  = max_stenosis_prob if is_stenosis else (1.0 - max_stenosis_prob)

elapsed = time.time() - t_start

# ── Save images ────────────────────────────────────────────────────────────────
url_orig    = save_img(orig_512,    'original.png')
url_sub     = save_img(sub_512,     'subtraction.png')
url_seg     = save_img(seg_512,     'segmentation.png')
url_overlay = save_img(overlay_img, 'overlay.png')

# ── Output JSON ────────────────────────────────────────────────────────────────
result = {
    'result_id':   opt.result_id,
    'elapsed_s':   round(elapsed, 2),
    'images': {
        'original':      url_orig,
        'subtraction':   url_sub,
        'segmentation':  url_seg,
        'overlay':       url_overlay,
    },
    'vessel_stats': {
        'vessel_pct':   round(vessel_pct, 2),
        'vessel_px':    vessel_px,
        'total_px':     total_px,
        'n_components': n_vessels,
        'max_area_px':  max_area,
        'mean_area_px': round(mean_area, 1),
    },
    'stenosis': {
        'detected':    is_stenosis,
        'confidence':  round(confidence * 100, 1),
        'max_prob':    round(max_stenosis_prob * 100, 1),
        'regions':     stenosis_regions[:10],
        'n_regions':   len(stenosis_regions),
        'severity':    'HIGH' if max_stenosis_prob > 0.85 else
                       'MODERATE' if max_stenosis_prob > 0.70 else
                       'LOW' if is_stenosis else 'NONE',
    }
}

print(json.dumps(result))
