import json
import re
from collections import Counter
from jiwer import cer, wer

# ==========================================
# 1. HÀM CHUẨN HÓA VĂN BẢN
# ==========================================
def clean_for_cer_wer(text):
    """
    Gộp văn bản thành 1 chuỗi liên tục để tính CER/WER:
    - Thay \\n bằng khoảng trắng
    - Xóa khoảng trắng thừa
    """
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_and_tokenize_for_f1(text):
    """
    Tách từ cho Bag-of-Words F1 (Bỏ qua dấu câu & hoa/thường)
    """
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()

# ==========================================
# 2. HÀM TÍNH BAG-OF-WORDS F1
# ==========================================
def calculate_bag_of_words_f1(preds_dict, keys_dict):
    total_tp = 0
    total_pred_len = 0
    total_key_len = 0

    for img_name, key_raw in keys_dict.items():
        if img_name not in preds_dict:
            continue

        pred_raw = preds_dict[img_name]

        key_words = clean_and_tokenize_for_f1(key_raw)
        pred_words = clean_and_tokenize_for_f1(pred_raw)

        key_counts = Counter(key_words)
        pred_counts = Counter(pred_words)

        common_words = key_counts & pred_counts
        tp = sum(common_words.values())

        total_tp += tp
        total_pred_len += len(pred_words)
        total_key_len += len(key_words)

    precision = total_tp / total_pred_len if total_pred_len > 0 else 0
    recall = total_tp / total_key_len if total_key_len > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

# ==========================================
# 3. MAIN EVALUATION
# ==========================================
def main():
    key_file = 'ground_truth.json'
    pred_file = 'Craft.json'

    # Load dữ liệu JSON
    with open(key_file, 'r', encoding='utf-8') as f:
        keys_data = json.load(f)

    with open(pred_file, 'r', encoding='utf-8') as f:
        preds_data = json.load(f)

    # Lấy danh sách văn bản tương ứng giữa Key và Prediction
    ground_truths = []
    hypotheses = []

    for img_name, key_text in keys_data.items():
        if img_name in preds_data:
            # Chuẩn hóa về 1 chuỗi để tính CER/WER
            gt_clean = clean_for_cer_wer(key_text)
            hyp_clean = clean_for_cer_wer(preds_data[img_name])

            ground_truths.append(gt_clean)
            hypotheses.append(hyp_clean)

    # 1. Tính CER và WER tổng thể
    overall_cer = cer(ground_truths, hypotheses)
    overall_wer = wer(ground_truths, hypotheses)

    # 2. Tính Word F1-Score (Kháng lỗi đảo dòng)
    p, r, f1 = calculate_bag_of_words_f1(preds_data, keys_data)

    # ==========================================
    # IN KẾT QUẢ
    # ==========================================
    print("==================================================")
    print("📊 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH OCR TOÀN DIỆN")
    print("==================================================")
    print(f"Tổng số ảnh đánh giá: {len(ground_truths)}\n")
    
    print("1️⃣ TRUYỀN THỐNG (Có xét đến thứ tự văn bản):")
    print(f"   • CER (Character Error Rate) : {overall_cer * 100:.2f}%  (Càng thấp càng tốt)")
    print(f"   • WER (Word Error Rate)      : {overall_wer * 100:.2f}%  (Càng thấp càng tốt)")
    print("\n2️⃣ BAG-OF-WORDS (Kháng lỗi lộn xộn/đảo thứ tự dòng):")
    print(f"   • Precision (Độ chính xác)  : {p * 100:.2f}%")
    print(f"   • Recall    (Độ phủ từ)     : {r * 100:.2f}%")
    print(f"   🎯 Word F1-Score            : {f1 * 100:.2f}%  (Càng cao càng tốt)")
    print("==================================================")

if __name__ == '__main__':
    main()