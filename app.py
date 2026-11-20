import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageChops, ImageOps
import cv2
import io
import time
import sys
import os

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepSA — Coronary Vessel AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── dark background ── */
.stApp { background: #0a0f1e; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1529 0%, #111827 100%);
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 20px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 3rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px;
}
.hero-sub { color: #94a3b8; font-size: 1.1rem; font-weight: 400; margin: 0; }
.hero-badge {
    display: inline-block; margin-top: 16px;
    background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4);
    color: #a5b4fc; padding: 4px 14px; border-radius: 99px; font-size: 0.8rem; font-weight: 500;
}

/* ── cards ── */
.card {
    background: linear-gradient(145deg, #111827, #0d1629);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}
.card-title {
    color: #e2e8f0; font-size: 0.85rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
}

/* ── metric chips ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }
.metric-chip {
    background: rgba(30,45,74,0.7);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 14px 20px;
    flex: 1; min-width: 120px;
    text-align: center;
}
.metric-chip .val { color: #818cf8; font-size: 1.6rem; font-weight: 700; line-height: 1; }
.metric-chip .lbl { color: #64748b; font-size: 0.72rem; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── output image panels ── */
.img-panel {
    background: #0d1629;
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    overflow: hidden;
}
.img-label {
    background: linear-gradient(90deg, #1e1b4b, #0d1629);
    color: #a5b4fc; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.8px; text-transform: uppercase;
    padding: 8px 16px; border-bottom: 1px solid #1e2d4a;
}

/* ── upload zone ── */
[data-testid="stFileUploader"] {
    background: rgba(13,22,41,0.6) !important;
    border: 2px dashed #312e81 !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1 !important; }

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 0.6rem 1.6rem !important; width: 100% !important;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px); }

/* ── download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: rgba(30,45,74,0.8) !important;
    color: #94a3b8 !important; border: 1px solid #1e3a5f !important;
    border-radius: 8px !important; font-size: 0.82rem !important;
    width: 100% !important;
}

/* ── selectbox / slider ── */
.stSelectbox > div > div { background: #111827 !important; border-color: #1e2d4a !important; color: #e2e8f0 !important; }
.stSlider > div { background: transparent !important; }
.stCheckbox > label > span { color: #94a3b8 !important; }

/* ── progress / spinner ── */
.stProgress > div > div { background: linear-gradient(90deg,#4f46e5,#7c3aed) !important; }

/* ── info / warning ── */
.stInfo { background: rgba(30,58,138,0.3) !important; border-color: #3b82f6 !important; color: #93c5fd !important; }

/* ── general text ── */
h1,h2,h3,h4,h5,h6,p,label,span { color: #e2e8f0; }
hr { border-color: #1e2d4a; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── model loading ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from models import UNet
from datasets import tophat
from utils import fusion_predict, make_mask, clear_mask

CKPT_OPTIONS = {
    "🏆 FS-CAD (Fine-Segmentation) — Recommended": "ckpt/fscad_36249.ckpt",
    "🔬 XCAD — Alternative Dataset":               "ckpt/xcad_4afe3.ckpt",
    "📦 Pre-trained Base (LM-CAD)":                "ckpt/pt_bc62a.ckpt",
}

@st.cache_resource(show_spinner=False)
def load_model(ckpt_name: str):
    path = os.path.join(BASE, ckpt_name)
    net = UNet(1, 1, 32, bilinear=True)
    ck = torch.load(path, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in ck["netE"].items()}
    net.load_state_dict(sd)
    net.eval()
    return net

def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def compute_vessel_stats(mask_img: Image.Image):
    arr = np.array(mask_img.convert("L"))
    binary = arr > 127
    total_px = binary.size
    vessel_px = binary.sum()
    pct = vessel_px / total_px * 100
    # connected components
    n_comp, lbl = cv2.connectedComponents(binary.astype(np.uint8))
    return {
        "vessel_pct": pct,
        "vessel_px": vessel_px,
        "components": max(0, n_comp - 1),
        "total_px": total_px,
    }

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🫀 DeepSA Controls")
    st.markdown("---")

    selected_ckpt_label = st.selectbox(
        "Model Checkpoint",
        list(CKPT_OPTIONS.keys()),
        index=0,
        help="FS-CAD model gives best vessel segmentation results."
    )
    ckpt_path = CKPT_OPTIONS[selected_ckpt_label]

    st.markdown("---")
    st.markdown("**⚙️ Inference Options**")

    auto_thresh = st.checkbox("Auto Threshold (recommended)", value=True,
        help="Fuses TopHat-enhanced and raw predictions for best quality.")
    multiangle  = st.checkbox("Multi-Angle TTA", value=True,
        help="Horizontal + vertical flip augmentation — reduces prediction variance.")
    pad_margin  = st.checkbox("Pad Margin", value=False,
        help="Adds 50px reflection padding to improve edge vessel detection.")

    st.markdown("---")
    st.markdown("**🎨 Post-Processing**")
    denoise_strength = st.slider("Denoise Strength", 1, 10, 4,
        help="fastNlMeansDenoising strength applied to subtraction output.")
    remove_size = st.slider("Min Object Size (px)", 100, 5000, 2000, 100,
        help="Remove connected components smaller than this pixel count.")
    hole_fill   = st.slider("Max Hole Fill (px)", 10, 300, 100, 10,
        help="Fill internal holes smaller than this area in the vessel mask.")

    st.markdown("---")
    st.markdown("""
    <div style='color:#475569;font-size:0.75rem;line-height:1.6'>
    <b style='color:#64748b'>DeepSA</b> — Deep Subtraction Angiography<br>
    CycleGAN pre-training + U-Net fine-tuning<br>
    512×512 · PyTorch · InstanceNorm
    </div>
    """, unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🫀 DeepSA</div>
  <p class="hero-sub">Deep Subtraction Angiography &nbsp;·&nbsp; AI-Powered Coronary Vessel Extraction &amp; Segmentation</p>
  <span class="hero-badge">✦ U-Net · CycleGAN · EMA Teacher · FS-CAD Fine-tuned</span>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ────────────────────────────────────────────────────────────────────
col_up, col_info = st.columns([3, 2], gap="large")

with col_up:
    st.markdown('<div class="card"><div class="card-title">📂 Upload Angiogram</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop or click to upload",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_info:
    st.markdown("""
    <div class="card">
    <div class="card-title">ℹ️ Supported Inputs</div>
    <ul style="color:#94a3b8;font-size:0.88rem;line-height:2;padding-left:18px;margin:0">
      <li>PNG / JPG / BMP / TIFF</li>
      <li>Grayscale or RGB (auto-converted)</li>
      <li>Any resolution (auto-resized to 512×512)</li>
      <li>Coronary angiogram frames</li>
      <li>DICOM screenshots / exported frames</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ── MAIN PROCESSING ───────────────────────────────────────────────────────────
if uploaded is not None:
    raw_img = Image.open(uploaded).convert("RGB")

    # show original
    st.markdown("---")
    col_orig, col_btn = st.columns([4, 1], gap="large")
    with col_orig:
        st.markdown('<div class="img-label">📷 ORIGINAL INPUT</div>', unsafe_allow_html=True)
        st.image(raw_img, use_container_width=True)
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        run = st.button("🚀 Run Analysis", use_container_width=True)
        if run:
            st.session_state["run"] = True

    # ── INFERENCE ────────────────────────────────────────────────────────────
    if st.session_state.get("run"):
        with st.spinner(""):
            prog = st.progress(0)
            status = st.empty()

            status.markdown('<p style="color:#818cf8;font-size:0.9rem">⚡ Loading model…</p>', unsafe_allow_html=True)
            net = load_model(ckpt_path)
            prog.progress(20)

            status.markdown('<p style="color:#818cf8;font-size:0.9rem">🔬 Preprocessing image…</p>', unsafe_allow_html=True)
            img_gray = raw_img.convert("L")

            SIZE = 512
            tfmc1 = T.Compose([
                T.Resize(SIZE),
                T.Lambda(lambda i: tophat(i, 50)),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
            tfmc2 = T.Compose([
                T.Resize(SIZE),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
            pad = 50 if pad_margin else 0
            prog.progress(35)

            status.markdown('<p style="color:#818cf8;font-size:0.9rem">🧠 Running neural network inference…</p>', unsafe_allow_html=True)
            t0 = time.time()

            if auto_thresh:
                x1 = tfmc1(img_gray)
                x2 = tfmc2(img_gray)
                _, out1 = fusion_predict(net, ["none"], x1, multiangle=multiangle,
                                         denoise=denoise_strength, size=SIZE,
                                         cutoff=0.4, pad=pad, netE=True)
                prog.progress(60)
                _, out2 = fusion_predict(net, ["none"], x2, multiangle=False,
                                         denoise=denoise_strength, size=SIZE,
                                         cutoff=0.4, pad=pad, netE=True)
                prog.progress(75)
                arr1 = np.array(out1.convert("L"))
                arr2 = np.array(out2.convert("L"))
                merged_arr = np.max(np.stack([arr1, arr2], axis=0), axis=0)
                out_merge = Image.fromarray(merged_arr).convert("RGB")
                mask_raw  = make_mask(out_merge, remove_size=remove_size,
                                      local_kernel=21, hole_max_size=hole_fill)
                out_adj   = T.functional.adjust_gamma(out_merge, 2)
                seg_img   = clear_mask(mask_raw)
                sub_img   = ImageChops.invert(out_adj)
                sub_t     = T.ToTensor()(sub_img)
                sub_t     = sub_t * (2 ** (-0.5))
                sub_img   = T.ToPILImage()(sub_t)
            else:
                x = tfmc1(img_gray)
                inp = x.unsqueeze(0)
                with torch.no_grad():
                    pred = net(inp)
                import torchvision
                sub_arr = torchvision.utils.make_grid(pred, normalize=True)
                sub_arr = (sub_arr.permute(1,2,0).cpu().numpy()*255).astype("uint8")
                sub_arr = cv2.fastNlMeansDenoising(sub_arr, None, denoise_strength, 7, 21)
                sub_img = Image.fromarray(sub_arr)
                sub_img = ImageOps.autocontrast(sub_img, cutoff=1)
                sub_img = T.functional.adjust_gamma(sub_img, 2)
                sub_img = ImageChops.invert(sub_img)
                sub_t   = T.ToTensor()(sub_img) * (2**(-0.5))
                sub_img = T.ToPILImage()(sub_t)
                import torch.nn.functional as F
                from skimage import morphology as skmorph
                seg_raw = torch.sign(pred)
                seg_np  = ((seg_raw.cpu().detach()+1)/2).numpy().astype(bool)
                seg_np  = skmorph.remove_small_objects(seg_np, remove_size)
                seg_arr = (seg_np*255).astype("uint8")
                seg_arr = torch.from_numpy(seg_arr/255)
                seg_img = T.ToPILImage()(seg_arr[0])

            elapsed = time.time() - t0
            prog.progress(90)

            status.markdown('<p style="color:#22c55e;font-size:0.9rem">✅ Computing vessel statistics…</p>', unsafe_allow_html=True)
            stats = compute_vessel_stats(seg_img)
            prog.progress(100)
            status.empty()

        # ── RESULTS ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="card-title" style="color:#a5b4fc;font-size:1rem;font-weight:700;margin-bottom:20px">📊 Analysis Results</div>', unsafe_allow_html=True)

        # metrics row
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-chip">
            <div class="val">{stats['vessel_pct']:.1f}%</div>
            <div class="lbl">Vessel Coverage</div>
          </div>
          <div class="metric-chip">
            <div class="val">{stats['vessel_px']:,}</div>
            <div class="lbl">Vessel Pixels</div>
          </div>
          <div class="metric-chip">
            <div class="val">{stats['components']}</div>
            <div class="lbl">Vessel Segments</div>
          </div>
          <div class="metric-chip">
            <div class="val">{elapsed:.2f}s</div>
            <div class="lbl">Inference Time</div>
          </div>
          <div class="metric-chip">
            <div class="val">512</div>
            <div class="lbl">Resolution (px)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # three output images
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown('<div class="img-label">📷 ORIGINAL</div>', unsafe_allow_html=True)
            orig_resized = raw_img.resize((512, 512), Image.BICUBIC)
            st.image(orig_resized, use_container_width=True)
            st.download_button("⬇ Download Original", pil_to_bytes(orig_resized),
                               "original.png", "image/png", use_container_width=True)

        with c2:
            st.markdown('<div class="img-label">✨ DEEP SUBTRACTION</div>', unsafe_allow_html=True)
            sub_disp = sub_img.convert("RGB").resize((512,512), Image.BICUBIC)
            st.image(sub_disp, use_container_width=True)
            st.download_button("⬇ Download Subtraction", pil_to_bytes(sub_disp),
                               "subtraction.png", "image/png", use_container_width=True)

        with c3:
            st.markdown('<div class="img-label">🩸 VESSEL SEGMENTATION</div>', unsafe_allow_html=True)
            seg_disp = seg_img.convert("RGB").resize((512,512), Image.BICUBIC)
            st.image(seg_disp, use_container_width=True)
            st.download_button("⬇ Download Mask", pil_to_bytes(seg_disp),
                               "vessel_mask.png", "image/png", use_container_width=True)

        # overlay panel
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card-title" style="color:#a5b4fc;font-size:0.9rem">🔴 Vessel Overlay</div>', unsafe_allow_html=True)
        orig_np  = np.array(orig_resized.convert("L"))
        orig_rgb = np.stack([orig_np]*3, axis=-1)
        mask_np  = np.array(seg_img.convert("L").resize((512,512))) > 127
        overlay  = orig_rgb.copy()
        overlay[mask_np, 0] = np.clip(overlay[mask_np, 0].astype(int) + 120, 0, 255)
        overlay[mask_np, 1] = np.clip(overlay[mask_np, 1].astype(int) - 40,  0, 255)
        overlay[mask_np, 2] = np.clip(overlay[mask_np, 2].astype(int) - 40,  0, 255)
        overlay_img = Image.fromarray(overlay.astype(np.uint8))
        co1, co2 = st.columns([2, 3])
        with co1:
            st.image(overlay_img, caption="Red = detected vessels", use_container_width=True)
        with co2:
            st.markdown(f"""
            <div class="card" style="height:100%">
            <div class="card-title">🔍 Detailed Statistics</div>
            <table style="width:100%;color:#94a3b8;font-size:0.88rem;border-collapse:collapse">
              <tr><td style="padding:8px 0;border-bottom:1px solid #1e2d4a">Model used</td>
                  <td style="color:#818cf8;text-align:right">{selected_ckpt_label.split('—')[0].strip()}</td></tr>
              <tr><td style="padding:8px 0;border-bottom:1px solid #1e2d4a">Auto threshold</td>
                  <td style="color:#818cf8;text-align:right">{'✅ Yes' if auto_thresh else '❌ No'}</td></tr>
              <tr><td style="padding:8px 0;border-bottom:1px solid #1e2d4a">Multi-angle TTA</td>
                  <td style="color:#818cf8;text-align:right">{'✅ Yes' if multiangle else '❌ No'}</td></tr>
              <tr><td style="padding:8px 0;border-bottom:1px solid #1e2d4a">Denoise strength</td>
                  <td style="color:#818cf8;text-align:right">{denoise_strength}</td></tr>
              <tr><td style="padding:8px 0;border-bottom:1px solid #1e2d4a">Min object size</td>
                  <td style="color:#818cf8;text-align:right">{remove_size} px</td></tr>
              <tr><td style="padding:8px 0;border-bottom:1px solid #1e2d4a">Vessel pixels</td>
                  <td style="color:#818cf8;text-align:right">{stats['vessel_px']:,}</td></tr>
              <tr><td style="padding:8px 0;border-bottom:1px solid #1e2d4a">Background pixels</td>
                  <td style="color:#818cf8;text-align:right">{stats['total_px']-stats['vessel_px']:,}</td></tr>
              <tr><td style="padding:8px 0">Inference time</td>
                  <td style="color:#22c55e;text-align:right">{elapsed:.3f} s</td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

        # reset
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Analyze Another Image"):
            st.session_state["run"] = False
            st.rerun()

else:
    # empty state
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;color:#334155">
      <div style="font-size:5rem;margin-bottom:24px">🩺</div>
      <div style="font-size:1.4rem;font-weight:600;color:#475569;margin-bottom:12px">
        Upload a Coronary Angiogram to Begin
      </div>
      <div style="color:#334155;font-size:0.9rem;max-width:500px;margin:0 auto;line-height:1.8">
        DeepSA will extract and segment coronary vessels using a<br>
        CycleGAN pre-trained + fine-tuned U-Net model.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='margin-top:48px;border-color:#1e2d4a'>
<div style='text-align:center;color:#334155;font-size:0.78rem;padding:12px 0 24px'>
  DeepSA &nbsp;·&nbsp; Deep Subtraction Angiography &nbsp;·&nbsp;
  U-Net + CycleGAN + EMA &nbsp;·&nbsp; For Research Use Only
</div>
""", unsafe_allow_html=True)
