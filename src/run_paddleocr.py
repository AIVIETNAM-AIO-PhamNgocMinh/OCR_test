import os

# 1. Tắt xung đột OpenMP/MKL giữa Paddle, PyTorch và OpenCV trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Giữ nguyên cấu hình Paddle của bạn
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import json
import time
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

# ---------------------------------------------------------
# 1. FIX LỖI ONEDNN / PIR
# ---------------------------------------------------------
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# ---------------------------------------------------------
# 2. KHỞI TẠO MODEL
# ---------------------------------------------------------
print("⏳ Đang khởi tạo PaddleOCR Detection & VietOCR Recognizer...")

paddle_det = PaddleOCR(
    use_textline_orientation=False,
    lang='vi',
    device='cpu',  # Đổi thành 'gpu' nếu dùng NVIDIA
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    text_detection_model_name='PP-OCRv5_server_det',
    text_det_thresh=0.2,
    text_det_box_thresh=0.5,
    text_det_unclip_ratio=2.2,  # Tăng unclip_ratio để bao trọn dấu tiếng Việt
)

vocr_config = Cfg.load_config_from_name('vgg_transformer')
vocr_config['device'] = 'cpu'  # Đổi thành 'cuda' nếu dùng GPU
vocr_config['predictor']['beamsearch'] = True  # Bật Beam Search để đoán dấu tiếng Việt chuẩn hơn
recognizer = Predictor(vocr_config)

print("✅ Đã khởi tạo xong Models!")

# ---------------------------------------------------------
# 3. CÁC HÀM XỬ LÝ HÌNH HỌC VÀ OCR
# ---------------------------------------------------------

def detect_boxes_original_scale(img_orig, det_model, max_side=1024):
    """
    Detect trên bản resize chuẩn và quy đổi trực tiếp về tọa độ ÁNH GỐC.
    """
    h_orig, w_orig = img_orig.shape[:2]
    
    # Tính toán scale sao cho cạnh lớn nhất = max_side
    scale = min(1.0, float(max_side) / max(h_orig, w_orig))
    if scale < 1.0:
        img_small = cv2.resize(img_orig, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_orig
        scale = 1.0

    results = det_model.predict(img_small)
    boxes = []
    
    if results and isinstance(results[0], dict):
        raw_boxes = results[0].get('dt_polys', None)
        if raw_boxes is not None:
            for poly in raw_boxes:
                # Map ngược về kích thước ảnh gốc
                xs = [p[0] / scale for p in poly]
                ys = [p[1] / scale for p in poly]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
        else:
            for box in results[0].get('rec_boxes', []):
                xmin, ymin, xmax, ymax = box
                boxes.append([xmin / scale, ymin / scale, xmax / scale, ymax / scale])
                
    return boxes


def preprocess_crop_for_vietocr(crop_rgb):
    """
    Tiền xử lý riêng trên từng CROP CHỮ để không làm nhiễu nền ảnh gốc:
    Tăng nét + Tăng độ phân giải cho các cụm chữ quá nhỏ.
    """
    h, w = crop_rgb.shape[:2]
    if h == 0 or w == 0:
        return crop_rgb

    # 1. Resize nếu chiều cao chữ quá nhỏ (< 32px)
    if h < 32:
        scale = 36.0 / float(h)
        new_w = max(int(w * scale), 10)
        crop_rgb = cv2.resize(crop_rgb, (new_w, 36), interpolation=cv2.INTER_CUBIC)

    # 2. Tăng nhẹ tương phản bằng CLAHE trên kênh L
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    return enhanced


def group_lines_by_overlap(items, overlap_ratio=0.45):
    """Gom các box chữ cùng dải Y nằm ngang."""
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


def split_row_into_columns(group, img_w, gap_multiplier=0.8, height_ratio_break=1.8, abs_gap_ratio=0.015):
    """Tách cột trục X nhạy bén."""
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


def build_final_lines(y_groups, img_w, gap_multiplier=0.8, height_ratio_break=1.8, abs_gap_ratio=0.015):
    """Sắp xếp văn bản theo đúng thứ tự đọc từ trên xuống dưới, trái sang phải."""
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
    """Lồng ghép toàn bộ quy trình xử lý cho 1 ảnh."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return ""

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]

    # 1. Detect Box trực tiếp từ ảnh gốc (chỉ downscale tạm thời để phát hiện nhanh)
    boxes = detect_boxes_original_scale(img_bgr, paddle_det, max_side=1024)
    if not boxes:
        return ""

    # 2. Crop với Padding thông minh
    crop_meta = []
    crop_images = []

    for box in boxes:
        xmin, ymin, xmax, ymax = [int(v) for v in box]
        h_box = ymax - ymin

        # Thêm Padding linh hoạt: pad dòng theo Y lớn hơn pad X để không cắt mất dấu tiếng Việt
        pad_y = max(4, int(h_box * 0.12))
        pad_x = max(3, int(h_box * 0.08))

        xmin_p = max(0, xmin - pad_x)
        ymin_p = max(0, ymin - pad_y)
        xmax_p = min(img_w, xmax + pad_x)
        ymax_p = min(img_h, ymax + pad_y)

        if xmax_p <= xmin_p or ymax_p <= ymin_p:
            continue

        crop = img_rgb[ymin_p:ymax_p, xmin_p:xmax_p]
        if crop.size == 0:
            continue

        # Tiền xử lý riêng biệt trên crop
        crop_enhanced = preprocess_crop_for_vietocr(crop)

        crop_images.append(Image.fromarray(crop_enhanced))
        crop_meta.append({
            'y': (ymin + ymax) / 2.0,
            'xmin': xmin,
            'xmax': xmax,
            'height': h_box
        })

    if not crop_images:
        return ""

    # 3. Batch Recognize bằng VietOCR
    batch_size = 16  # Chia nhỏ 16 ảnh crop/lần
    texts = []
    for i in range(0, len(crop_images), batch_size):
        chunk = crop_images[i : i + batch_size]
        chunk_texts = recognizer.predict_batch(chunk)
        texts.extend(chunk_texts)

    items = []
    for text, meta in zip(texts, crop_meta):
        clean_t = text.strip()
        if not clean_t or (len(clean_t) == 1 and not clean_t.isalnum()):
            continue
        items.append({**meta, "text": clean_t})

    # 4. Phân nhóm dòng Y & Tách cột X
    y_groups = group_lines_by_overlap(items, overlap_ratio=0.45)
    lines = build_final_lines(
        y_groups, img_w,
        gap_multiplier=0.8,
        height_ratio_break=1.8,
        abs_gap_ratio=0.015,
    )

    return "\n".join(lines)


# ---------------------------------------------------------
# 4. CHẠY BATCH TRÊN THƯ MỤC
# ---------------------------------------------------------
IMAGE_DIR = os.path.join("Dataset", "Images")
OUTPUT_FILE = os.path.join("Dataset", "draft_labels_v2.json")

if not os.path.exists(IMAGE_DIR):
    print(f"❌ Không tìm thấy thư mục ảnh tại: {IMAGE_DIR}")
    exit()

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)])

if not image_files:
    print(f"⚠️ Thư mục '{IMAGE_DIR}' không có ảnh nào!")
    exit()

print(f"📸 Tìm thấy {len(image_files)} ảnh. Bắt đầu xử lý...")

draft_data = {}
start_time = time.time()

for img_name in tqdm(image_files, desc="Đang OCR"):
    img_path = os.path.join(IMAGE_DIR, img_name)
    print(f"\nĐang xử lý: {img_name}") # Un-comment dòng này để xem ảnh nào bị treo
    try:
        full_text = process_single_image(img_path)
        draft_data[img_name] = full_text
    except Exception as e:
        print(f"\n❌ Lỗi ảnh {img_name}: {e}")
        draft_data[img_name] = ""

total_time = time.time() - start_time

# ---------------------------------------------------------
# 5. GHI KẾT QUẢ JSON
# ---------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(draft_data, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 50)
print(f"✅ ĐÃ XỬ LÝ XONG {len(image_files)} ẢNH!")
print(f"⏱️ Tổng thời gian: {total_time:.2f} giây")
print(f"📂 Kết quả lưu tại: {OUTPUT_FILE}")
print("=" * 50)