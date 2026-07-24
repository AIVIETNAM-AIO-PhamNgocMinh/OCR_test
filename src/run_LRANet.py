import sys
import os
import json
import time
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# ================= 1. CHUẨN BỊ ĐƯỜNG DẪN & WORKING DIRECTORY =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chuyển Working Directory vào trong thư mục LRANet-PP
os.chdir(os.path.join(BASE_DIR, 'LRANet-PP'))

# Thêm đường dẫn vào sys.path
sys.path.insert(0, '.')
sys.path.insert(0, './mmocr/core/evaluation/evaluation_e2e')

# --- MONKEY PATCH MMCV MODULATED DEFORM CONV BEFORE ANY OTHER IMPORT ---
import torchvision.ops
import mmcv.ops.modulated_deform_conv
import mmcv.ops

def patched_modulated_deform_conv2d(x, offset, mask, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, deform_groups=1):
    def _pair(v):
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return (v, v)
    return torchvision.ops.deform_conv2d(
        input=x, offset=offset, weight=weight, bias=bias,
        stride=_pair(stride), padding=_pair(padding),
        dilation=_pair(dilation), mask=mask
    )

mmcv.ops.modulated_deform_conv.modulated_deform_conv2d = patched_modulated_deform_conv2d
mmcv.ops.modulated_deform_conv2d = patched_modulated_deform_conv2d

# Import để register các class model của MMOCR vào MMDetection Registry
import mmocr.models  # noqa: F401

import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, DataContainer

# --- MONKEY PATCH TPSAlign TO USE CHUNKED FORWARD FOR MEMORY EFFICIENCY ---
from mmocr.models.textend2end.utils.tps_align import TPSAlign

def chunked_tpsalign_forward(self, feature_map, grids, batch_idx, texts):
    grids = grids.detach()
    grid_reshaped = grids.view(-1, self.grid_size[0], self.grid_size[1], 2)
    num_rois = grid_reshaped.shape[0]

    if num_rois == 0:
        feats = feature_map.new_zeros((0, feature_map.shape[1], self.grid_size[0], self.grid_size[1]))
        return feats, texts

    chunk_size = 1  # Process one ROI at a time to prevent CPU OOM
    feats_list = []
    for i in range(0, num_rois, chunk_size):
        grid_chunk = grid_reshaped[i: i + chunk_size]
        batch_idx_chunk = batch_idx[i: i + chunk_size].long()
        feat_chunk = F.grid_sample(
            feature_map[batch_idx_chunk], grid_chunk,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        feats_list.append(feat_chunk)

    return torch.cat(feats_list, dim=0), texts

TPSAlign.forward = chunked_tpsalign_forward

# ================= 2. CONFIG PATHS =================
CONFIG_PATH = os.path.join(BASE_DIR, 'LRANet-PP/configs/lranet_pp/lranet_pp_totaltext.py')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'LRANet-PP/totaltext_final.pth')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ================= 3. PIPELINE SELECTION =================
PIPELINE = 'lranet_vietocr'

# ================= 4. KHỞI TẠO MODEL =================
print("Dang khoi tao LRANet model...")
cfg = Config.fromfile(CONFIG_PATH)
if hasattr(cfg.model, 'pretrained'):
    cfg.model.pretrained = None

lranet_model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(lranet_model, CHECKPOINT_PATH, map_location=device)
lranet_model.to(device)
lranet_model.eval()
print("  -> LRANet da san sang!")

lranet_test_pipeline = Compose(cfg.data.test.pipeline)

if PIPELINE == 'lranet_vietocr':
    print("Dang khoi tao VietOCR recognizer...")
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    
    vocr_config = Cfg.load_config_from_name('vgg_transformer') 
    vocr_config['device'] = 'cpu'
    vocr_config['predictor']['beamsearch'] = True  # Bật Beam Search để tăng độ chính xác
    vietocr_recognizer = Predictor(vocr_config)
    print("  -> VietOCR da san sang!")
else:
    vietocr_recognizer = None

print(f"Pipeline: {PIPELINE}\n")

# ================= 5. UTILITY FUNCTIONS & PREPROCESSING =================

def preprocess_crop_for_vietocr(crop_rgb):
    """
    Tiền xử lý ảnh chữ trước khi đưa vào VietOCR:
    1. CLAHE (Tăng tương phản)
    2. Unsharp Masking (Làm nét dấu tiếng Việt)
    3. Upscale chữ nhỏ
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return crop_rgb

    # 1. Tăng tương phản bằng CLAHE
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # 2. Làm sắc nét viền chữ và dấu tiếng Việt
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    unsharp = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

    # 3. Upscale nếu chữ quá bé (height < 32px)
    h, w = unsharp.shape[:2]
    if h < 32:
        scale = 36.0 / max(h, 1)
        new_w = max(int(w * scale), 10)
        unsharp = cv2.resize(unsharp, (new_w, 36), interpolation=cv2.INTER_CUBIC)

    return unsharp


def run_lranet_inference(img_path):
    """Chay LRANet inference tren mot anh, tra ve ket qua raw."""
    data = lranet_test_pipeline(dict(img_info=dict(filename=img_path), img_prefix=None))
    data = collate([data], samples_per_gpu=1)

    # Unpack DataContainers
    if isinstance(data['img'], list):
        data['img'] = [x.data[0] if isinstance(x, DataContainer) else x for x in data['img']]
    elif isinstance(data['img'], DataContainer):
        data['img'] = data['img'].data
    if not isinstance(data['img'], list):
        data['img'] = [data['img']]
    data['img'] = [img.to(device) if isinstance(img, torch.Tensor) else img for img in data['img']]

    if isinstance(data['img_metas'], list):
        if len(data['img_metas']) > 0 and isinstance(data['img_metas'][0], DataContainer):
            data['img_metas'] = data['img_metas'][0].data
    elif isinstance(data['img_metas'], DataContainer):
        data['img_metas'] = data['img_metas'].data

    with torch.no_grad():
        result = lranet_model(return_loss=False, rescale=True, **data)

    if isinstance(result, list):
        result = result[0]
    return result


def get_lranet_boxes(result):
    """
    Lay danh sach bounding box tu ket qua LRANet.
    Tra ve list of [xmin, ymin, xmax, ymax].
    """
    boxes = []
    boundaries = result.get('boundary_result', [])
    for b in boundaries:
        coords = np.array(b[:-1]).reshape(-1, 2)
        xmin = float(coords[:, 0].min())
        ymin = float(coords[:, 1].min())
        xmax = float(coords[:, 0].max())
        ymax = float(coords[:, 1].max())
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def group_lines_by_overlap(items, overlap_ratio=0.5):
    """Gom cac box chu cung dai Y nam ngang."""
    items = sorted(items, key=lambda i: i['y'])
    lines = []

    for item in items:
        h = item['height']
        i_ymin = item['y'] - h / 2
        i_ymax = item['y'] + h / 2

        best_line = None
        best_overlap = 0
        for line in lines:
            overlap = min(i_ymax, line['ymax']) - max(i_ymin, line['ymin'])
            min_h = min(h, line['ymax'] - line['ymin'])
            if overlap > 0 and overlap > best_overlap and overlap > min_h * overlap_ratio:
                best_overlap = overlap
                best_line = line

        if best_line is not None:
            best_line['items'].append(item)
            best_line['ymin'] = min(best_line['ymin'], i_ymin)
            best_line['ymax'] = max(best_line['ymax'], i_ymax)
        else:
            lines.append({'items': [item], 'ymin': i_ymin, 'ymax': i_ymax})

    lines.sort(key=lambda l: sum(i['y'] for i in l['items']) / len(l['items']))
    return [l['items'] for l in lines]


def split_row_into_columns(group, img_w, gap_multiplier=1.0, height_ratio_break=1.8, abs_gap_ratio=0.02):
    """Tach cot truc X: tach rieng neu khoang cach X qua lon."""
    group = sorted(group, key=lambda i: i['xmin'])
    clusters = [[group[0]]]
    abs_gap_floor = img_w * abs_gap_ratio

    for item in group[1:]:
        prev = clusters[-1][-1]
        gap_x = item['xmin'] - prev['xmax']
        ref_h = min(item['height'], prev['height'])
        h_ratio = max(item['height'], prev['height']) / max(ref_h, 1e-6)

        should_split = (
            gap_x > (ref_h * gap_multiplier)
            or h_ratio > height_ratio_break
            or gap_x > abs_gap_floor
        )

        if should_split:
            clusters.append([item])
        else:
            clusters[-1].append(item)

    return clusters


def build_final_lines(y_groups, img_w, gap_multiplier=1.0, height_ratio_break=1.8, abs_gap_ratio=0.02):
    """Xuat van ban da ngat dong chuan xac theo thu tu tu tren xuong duoi."""
    final_lines = []
    for group in y_groups:
        clusters = split_row_into_columns(
            group, img_w,
            gap_multiplier=gap_multiplier,
            height_ratio_break=height_ratio_break,
            abs_gap_ratio=abs_gap_ratio,
        )
        for cluster in clusters:
            cluster.sort(key=lambda i: i['xmin'])
            text = " ".join(i['text'] for i in cluster)
            final_lines.append({
                'text': text,
                'ymin': min(i['y'] for i in cluster)
            })

    final_lines.sort(key=lambda l: l['ymin'])
    return [l['text'] for l in final_lines]


def process_single_image(img_path):
    """
    Xu ly tron goi 1 anh:
    - LRANet detect -> lay bbox
    - VietOCR recognize
    - Gom nhom dong Y, tach cot X
    - Tra ve text da sap xep
    """
    result = run_lranet_inference(img_path)

    if PIPELINE == 'lranet_vietocr':
        boxes = get_lranet_boxes(result)
        if not boxes:
            return ""

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            return ""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        crop_meta = []
        crop_images = []
        pad = 3

        for box in boxes:
            xmin, ymin, xmax, ymax = [int(v) for v in box]
            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(img_w, xmax + pad)
            ymax = min(img_h, ymax + pad)

            if xmax <= xmin or ymax <= ymin:
                continue

            crop = img_rgb[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                continue

            # Tiền xử lý nâng cao cho Crop
            crop_enhanced = preprocess_crop_for_vietocr(crop)

            crop_images.append(Image.fromarray(crop_enhanced))
            crop_meta.append({
                'y': (ymin + ymax) / 2.0,
                'xmin': xmin,
                'xmax': xmax,
                'height': ymax - ymin,
            })

        if not crop_images:
            return ""

        # Nhận dạng bằng VietOCR
        texts = vietocr_recognizer.predict_batch(crop_images)

        items = []
        for text, meta in zip(texts, crop_meta):
            clean_t = text.strip()
            if not clean_t or (len(clean_t) == 1 and not clean_t.isalnum()):
                continue
            items.append({**meta, 'text': clean_t})

        if not items:
            return ""

    else:
        strs = result.get('strs', [])
        boundaries = result.get('boundary_result', [])

        if not strs:
            return ""

        img_bgr = cv2.imread(img_path)
        img_h, img_w = img_bgr.shape[:2] if img_bgr is not None else (1080, 1920)

        items = []
        for text, boundary in zip(strs, boundaries):
            clean_t = text.strip()
            if not clean_t:
                continue
            coords = np.array(boundary[:-1]).reshape(-1, 2)
            xmin = float(coords[:, 0].min())
            ymin = float(coords[:, 1].min())
            xmax = float(coords[:, 0].max())
            ymax = float(coords[:, 1].max())
            items.append({
                'text': clean_t,
                'y': (ymin + ymax) / 2.0,
                'xmin': xmin,
                'xmax': xmax,
                'height': ymax - ymin,
            })

        if not items:
            return ""

    img_w_use = img_w

    y_groups = group_lines_by_overlap(items, overlap_ratio=0.5)
    lines = build_final_lines(
        y_groups, img_w_use,
        gap_multiplier=1.0,
        height_ratio_break=1.8,
        abs_gap_ratio=0.02,
    )

    return "\n".join(lines)


# ================= 6. XỬ LÝ TOÀN BỘ FOLDER ẢNH =================
IMAGE_DIR = os.path.join(BASE_DIR, 'Dataset', 'Images')
OUTPUT_FILE = os.path.join(BASE_DIR, 'Dataset', f'draft_labels_lranet_{PIPELINE}.json')

if not os.path.exists(IMAGE_DIR):
    print(f"❌ Khong tim thay thu muc anh tai: {IMAGE_DIR}")
    sys.exit(1)

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)])

if not image_files:
    print(f"❌ Thu muc '{IMAGE_DIR}' khong co anh nao!")
    sys.exit(1)

print(f"🚀 Tim thay {len(image_files)} anh. Bat dau xu ly batch...\n")

draft_data = {}
start_time = time.time()

# Tiến hành loop chạy qua từng ảnh với thanh tiến trình tqdm
for img_name in tqdm(image_files, desc="Dang OCR Batch"):
    img_path = os.path.join(IMAGE_DIR, img_name)
    try:
        full_text = process_single_image(img_path)
        draft_data[img_name] = full_text
    except Exception as e:
        print(f"\n⚠️ Loi tai anh {img_name}: {e}")
        draft_data[img_name] = ""

total_time = time.time() - start_time

# ================= 7. GHI KẾT QUẢ RA FILE JSON =================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(draft_data, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print(f"🎉 ĐÃ XỬ LÝ XONG TOÀN BỘ {len(image_files)} ẢNH!")
print(f"⏱️ Tổng thời gian chạy: {total_time:.2f} giây")
print(f"📂 Kết quả đã được xuất ra file: {OUTPUT_FILE}")
print("=" * 50)