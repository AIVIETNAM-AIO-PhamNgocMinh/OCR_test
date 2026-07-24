import os
import json
import time
import traceback
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
try:
    from torchvision.models.vgg import model_urls
except ImportError:
    model_urls = {
        'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c47f32d.pth'
    }

# Import CRAFT và VietOCR
from craft_text_detector import Craft
import craft_text_detector.craft_utils as craft_utils
import craft_text_detector.predict as craft_predict
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# ---------------------------------------------------------
# 0. FIX TẬN GỐC: thư viện craft-text-detector dùng np.array() trên các
#    list "ragged" (polygon có số đỉnh khác nhau) ở NHIỀU chỗ khác nhau
#    trong craft_utils.py và predict.py — mỗi ảnh có thể kích hoạt một
#    dòng khác nhau tuỳ số lượng/hình dạng polygon detect được. Đây là
#    hành vi NumPy < 1.24 từng cho phép (tự fallback dtype=object) nhưng
#    NumPy >= 1.24 / 2.x cấm hẳn và raise ValueError.
#
#    Thay vì vá từng dòng một mỗi khi gặp lỗi mới, ta thay thế np.array
#    bên trong 2 module này bằng bản "an toàn": nếu gặp đúng lỗi ragged
#    array thì tự fallback dtype=object (giữ đúng hành vi NumPy cũ),
#    các lỗi khác vẫn raise bình thường để không che giấu bug thật.
# ---------------------------------------------------------

class _SafeNumpyProxy:
    """Proxy cho module numpy: mọi thuộc tính đều trỏ thẳng tới numpy thật,
    riêng np.array() được bọc lại để không crash khi gặp ragged array."""
    def __getattr__(self, name):
        return getattr(np, name)

    def array(self, obj, *args, **kwargs):
        try:
            return np.array(obj, *args, **kwargs)
        except ValueError as e:
            if "inhomogeneous" in str(e):
                kwargs.pop("dtype", None)
                return np.array(obj, dtype=object, *args, **kwargs)
            raise


_safe_np = _SafeNumpyProxy()
craft_utils.np = _safe_np
craft_predict.np = _safe_np

print("🩹 Đã vá np.array() trong craft_utils.py và predict.py (fix tận gốc lỗi ragged array).")

# ---------------------------------------------------------
# 1. KHỞI TẠO MODEL
# ---------------------------------------------------------
print("⏳ Đang khởi tạo CRAFT Detector & VietOCR Predictor...")

# Khởi tạo CRAFT — đã tune threshold để tăng khả năng bắt chữ nhỏ/mờ
craft_detector = Craft(
    output_dir=None,
    crop_type="poly",       # polygon để bắt nét chữ cong/nghiêng
    cuda=False,              # đổi thành True nếu dùng GPU NVIDIA
    rectify=True,            # bật nắn thẳng chữ (TPS / Perspective Transform)
    text_threshold=0.65,     # mặc định 0.7 — giảm nhẹ để bắt chữ mờ/nhòe hơn
    link_threshold=0.35,     # mặc định 0.4 — giảm để nối các ký tự rời rạc tốt hơn
    low_text=0.35,           # mặc định 0.4 — giảm để giữ lại vùng text có độ tin cậy thấp hơn
    long_size=1280,          # cạnh dài tối đa khi CRAFT resize nội bộ — tăng nếu ảnh có chữ rất nhỏ
)

# Khởi tạo VietOCR
vocr_config = Cfg.load_config_from_name('vgg_transformer')
vocr_config['device'] = 'cpu'  # đổi thành 'cuda' nếu dùng GPU
vocr_config['predictor']['beamsearch'] = True  # giúp nhận diện chính xác dấu tiếng Việt
vietocr_predictor = Predictor(vocr_config)

print("✅ Đã khởi tạo xong Models!")

# ---------------------------------------------------------
# 2. TIỀN XỬ LÝ ẢNH TRƯỚC KHI ĐƯA VÀO CRAFT
# ---------------------------------------------------------

def preprocess_image(img_bgr):
    """Tăng tương phản cục bộ (CLAHE) — giúp CRAFT bắt biên chữ rõ hơn,
    đặc biệt với ảnh chụp màn hình/video có nén JPEG, tương phản thấp."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def adaptive_resize(img_bgr):
    """Upscale thích ứng: chỉ scale mạnh nếu ảnh gốc có độ phân giải thấp,
    tránh phóng to vô ích làm chậm CRAFT trên ảnh đã đủ lớn."""
    h, w = img_bgr.shape[:2]
    if h < 720:
        scale = 2.0
    elif h < 1080:
        scale = 1.5
    else:
        scale = 1.0
    if scale == 1.0:
        return img_bgr
    return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------
# 3. HÀM CHUẨN HOÁ POLYGON (FIX LỖI inhomogeneous shape)
# ---------------------------------------------------------

def safe_poly_to_points(poly):
    """
    Chuẩn hoá polygon trả về từ CRAFT thành mảng (N, 2) đồng nhất.

    Lý do cần hàm này: CRAFT (dựa trên contour của OpenCV) đôi khi trả về
    các điểm trong CÙNG 1 polygon với shape lồng nhau khác nhau — có điểm
    dạng [x, y] (shape (2,)), có điểm dạng [[x, y]] (shape (1,2), do chưa
    được squeeze hết từ contour gốc). Khi ép thẳng np.array(poly) trên dữ
    liệu lẫn lộn này, NumPy báo lỗi "inhomogeneous shape" vì không thể suy
    ra 1 shape đồng nhất cho toàn bộ mảng.

    Hàm này duyệt từng điểm, "làm phẳng" (flatten) về đúng dạng [x, y],
    bỏ qua điểm lỗi/thiếu toạ độ thay vì làm crash cả polygon.
    """
    pts = []
    try:
        for p in poly:
            arr = np.asarray(p, dtype=np.float32).reshape(-1)
            if arr.size >= 2:
                pts.append([float(arr[0]), float(arr[1])])
    except Exception:
        return None

    if len(pts) < 3:
        return None

    return np.array(pts, dtype=np.float32)


# ---------------------------------------------------------
# 4. HÀM XỬ LÝ HÌNH HỌC VÀ TPS (RECTIFICATION)
# ---------------------------------------------------------

def apply_tps_warp(img, poly_pts, target_height=48):
    """
    Nắn thẳng polygon (đã chuẩn hoá, shape (N,2)) về hình chữ nhật chuẩn
    bằng Perspective Transform.
    """
    pts = poly_pts

    # Nếu polygon không đúng 4 đỉnh (chữ cong/nhiều đỉnh), lấy Bounding Box
    if pts.shape[0] != 4:
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)

    # Xác định 4 góc: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    if max_width <= 2 or max_height <= 2:
        return None

    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    src_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, matrix, (max_width, max_height))

    scale = target_height / float(max_height)
    new_width = max(int(max_width * scale), 16)
    warped_resized = cv2.resize(warped, (new_width, target_height), interpolation=cv2.INTER_CUBIC)

    return warped_resized


def group_lines_and_sort(items, overlap_ratio=0.45):
    """Gom nhóm các box theo dải Y nằm ngang và sắp xếp theo thứ tự đọc từ trên xuống, trái sang phải."""
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

    final_lines = []
    for line in lines:
        sorted_line = sorted(line['items'], key=lambda i: i['xmin'])
        text_line = " ".join(i['text'] for i in sorted_line)
        final_lines.append(text_line)

    return final_lines


# ---------------------------------------------------------
# 5. HÀM XỬ LÝ TRỌN GÓI CHO 1 ẢNH
# ---------------------------------------------------------

def normalize_image_format(img_bgr):
    """
    Ép ảnh về đúng 3 kênh BGR, dtype uint8 — phòng trường hợp ảnh gốc là
    grayscale (1 kênh), có kênh alpha (4 kênh), hoặc dtype khác uint8
    (ví dụ float), những thứ có thể khiến bước resize/pad nội bộ của
    CRAFT tạo ra mảng không đồng nhất.
    """
    if img_bgr is None:
        return None

    if img_bgr.dtype != np.uint8:
        img_bgr = cv2.normalize(img_bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    elif img_bgr.shape[2] == 1:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    return img_bgr


def process_single_image(img_path, keep_temp=False):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_bgr = normalize_image_format(img_bgr)
    if img_bgr is None:
        return ""

    # --- Tiền xử lý: CLAHE + upscale thích ứng ---
    img_bgr = preprocess_image(img_bgr)
    img_bgr = adaptive_resize(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # CRAFT trong bản craft-text-detector đang dùng cần path ảnh -> ghi ảnh
    # đã tiền xử lý ra file tạm để toạ độ polygon khớp với img_rgb dùng để crop
    temp_path = img_path + ".__preproc_temp.jpg"
    cv2.imwrite(temp_path, img_bgr)

    try:
        prediction_result = craft_detector.detect_text(temp_path)
    finally:
        if not keep_temp and os.path.exists(temp_path):
            os.remove(temp_path)

    polygons = prediction_result.get("polys", [])
    if len(polygons) == 0:
        return ""

    crop_images = []
    crop_meta = []

    for poly in polygons:
        # --- CHUẨN HOÁ POLYGON (fix lỗi inhomogeneous shape) ---
        poly_pts = safe_poly_to_points(poly)
        if poly_pts is None:
            continue

        xmin = float(np.min(poly_pts[:, 0]))
        ymin = float(np.min(poly_pts[:, 1]))
        ymax = float(np.max(poly_pts[:, 1]))

        # Lọc polygon quá nhỏ / suy biến (nhiễu, không phải chữ thật)
        if (ymax - ymin) < 4:
            continue

        warped_crop = apply_tps_warp(img_rgb, poly_pts, target_height=48)
        if warped_crop is None or warped_crop.size == 0:
            continue

        crop_images.append(Image.fromarray(warped_crop))
        crop_meta.append({
            'y': (ymin + ymax) / 2.0,
            'xmin': xmin,
            'height': ymax - ymin
        })

    if not crop_images:
        return ""

    # --- Batch Recognize bằng VietOCR ---
    batch_size = 16
    texts = []
    for i in range(0, len(crop_images), batch_size):
        chunk = crop_images[i: i + batch_size]
        chunk_texts = vietocr_predictor.predict_batch(chunk)
        texts.extend(chunk_texts)

    items = []
    for text, meta in zip(texts, crop_meta):
        clean_t = text.strip()
        if not clean_t or (len(clean_t) == 1 and not clean_t.isalnum()):
            continue
        items.append({**meta, 'text': clean_t})

    if not items:
        return ""

    lines = group_lines_and_sort(items, overlap_ratio=0.45)
    return "\n".join(lines)


# ---------------------------------------------------------
# 6. QUẢN LÝ THƯ MỤC VÀ CHẠY TIẾN TRÌNH
# ---------------------------------------------------------

if __name__ == "__main__":
    IMAGE_DIR = os.path.join("Dataset", "Images")
    OUTPUT_FILE = os.path.join("Dataset", "craft_vietocr_results.json")

    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Không tìm thấy thư mục ảnh tại: {IMAGE_DIR}")
        exit()

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)])

    if not image_files:
        print(f"⚠️ Thư mục '{IMAGE_DIR}' không có ảnh nào!")
        exit()

    print(f"📸 Tìm thấy {len(image_files)} ảnh. Bắt đầu xử lý...")

    results = {}
    start_time = time.time()

    for img_name in tqdm(image_files, desc="Đang OCR"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        try:
            full_text = process_single_image(img_path)
            results[img_name] = full_text
        except Exception as e:
            print(f"\n❌ Lỗi ảnh {img_name}: {e}")
            traceback.print_exc()  # in đầy đủ stack trace để xác định chính xác dòng gây lỗi
            results[img_name] = ""

    total_time = time.time() - start_time

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"✅ ĐÃ XỬ LÝ XONG {len(image_files)} ẢNH!")
    print(f"⏱️ Tổng thời gian: {total_time:.2f} giây")
    print(f"📂 Kết quả lưu tại: {OUTPUT_FILE}")
    print("=" * 50)