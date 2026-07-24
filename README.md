# So sánh các cấu trúc OCR tiếng Việt

## 1. Các mô hình sử dụng

**Mô hình nhận diện text box:**
- PaddleOCR
- CRAFT + TPS
- LRNet++

**Mô hình nhận diện văn bản tiếng Việt:** VietOCR

---

## 2. Metrics

* **WER (Word Error Rate):** Tỷ lệ lỗi ở cấp độ từ.
* **CER (Character Error Rate):** Tỷ lệ lỗi ở cấp độ ký tự.

---

## 3. Cấu trúc Thư mục Project

```text
.
├── Dataset/
│   ├── Images/                 # Thư mục chứa các ảnh input (.jpg, .png, ...)
│   └── ground_truth.json       # File chứa nhãn chuẩn dạng JSON cho từng ảnh
├── Result/
│   ├── Craft_result.json       # Kết quả OCR đầu ra của pipeline CRAFT + TPS + VietOCR
│   ├── LRANet_result.json      # Kết quả OCR đầu ra của pipeline LRNet++ + VietOCR
│   └── PaddleOCR_result.json   # Kết quả OCR đầu ra của pipeline PaddleOCR + VietOCR
├── src/
│   ├── metrics.py              # Script tính toán chỉ số WER và CER so sánh các file Result
│   ├── run_craft.py            # Script thực thi pipeline CRAFT + TPS + VietOCR
│   ├── run_LRANet.py           # Script thực thi pipeline LRNet++ + VietOCR
│   └── run_paddleocr.py        # Script thực thi pipeline PaddleOCR + VietOCR
└── README.md                   # Tài liệu hướng dẫn dự án
```

---

## 4. Hướng dẫn cài đặt & Môi trường

### 4.1. Khởi tạo môi trường Conda
Nên tạo một môi trường Python từ 3.9 đến 3.10 để đảm bảo tính tương thích tốt nhất cho các thư viện DL:

```bash
conda create -n ocr_bench python=3.10 -y
conda activate ocr_bench
```

### 4.2. Cài đặt PyTorch & Torchvision
Cài đặt PyTorch tương thích (bản CPU hoặc CUDA 11.8/12.1):

```bash
# Đối với GPU (CUDA 11.8)
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Hoặc đối với CPU
pip install torch torchvision
```

### 4.3. Cài đặt các thư viện OCR & Dependency

```bash
# 1. Cài đặt VietOCR và OpenCV
pip install vietocr opencv-python pillow numpy tqdm

# 2. Cài đặt PaddleOCR & PaddlePaddle
pip install paddlepaddle paddleocr

# 3. Cài đặt CRAFT Text Detector
pip install craft-text-detector

# 4. Cài đặt thư viện tính độ đo WER/CER
pip install jiwer
```

---

## 5. Hướng dẫn Sử dụng

### Bước 1: Chuẩn bị dữ liệu
Đặt các ảnh thử nghiệm vào thư mục `Dataset/Images/` và chuẩn bị file `Dataset/ground_truth.json` theo định dạng:
```json
{
  "001.jpg": "Văn bản mẫu tiếng Việt",
  "002.jpg": "Công nghệ thông tin"
}
```

### Bước 2: Chạy các pipeline OCR
Lần lượt thực thi các file chạy trong thư mục `src/`:

```bash
# Chạy pipeline CRAFT
python src/run_craft.py

# Chạy pipeline PaddleOCR
python src/run_paddleocr.py

# Chạy pipeline LRNet++
python src/run_LRANet.py
```

### Bước 3: Tính toán & So sánh Metrics (WER / CER)
Chạy script `metrics.py` để xuất báo cáo đánh giá:

```bash
python src/metrics.py
```
