# Đánh giá hiệu năng

## 1. Kết quả thực nghiệm

| Pipeline | CER (↓) | WER (↓) | Precision (↑) | Recall (↑) | Word F1-Score (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LRNet++ + VietOCR** | **29.21%** | **47.19%** | 71.18% | **74.01%** | 72.57% |
| **PaddleOCR + VietOCR** | 32.39% | 48.44% | **74.24%** | 72.62% | 73.42% |
| **CRAFT + TPS + VietOCR** | 32.83% | 48.65% | 73.06% | 74.80% | **73.92%** |

*(Ghi chú: `↓` Càng thấp càng tốt | `↑` Càng cao càng tốt)*

---

## 2. Phân tích

Từ kết quả thực nghiệm trên, ta rút ra một số nhận xét:

1. **CER & WER - Có xét đến thứ tự văn bản:**
   * **LRNet++** đạt **CER tốt nhất (29.21%)** và **WER tốt nhất (47.19%)**. Nguyên nhân là LRNet++ phân đoạn dải dòng (Text Line) vô cùng mạnh mẽ, bắt trọn vẹn các vùng chữ uốn éo hoặc dị dạng mà hai mô hình còn lại dễ bị cắt lẹm.
   * Tuy nhiên, LRNet++ lại gặp điểm yếu về **thứ tự đọc (Line Reading Order)**: các dải dòng cắt ra từ LRNet++ đôi khi bị sắp xếp lộn xộn/đảo vị trí dòng trước - dòng sau. Khi tính toán theo thứ tự chuỗi truyền thống, việc đảo dòng sẽ bị phạt điểm nặng ở cấp độ từ/câu.

2. **Bag-of-Words (Precision, Recall & F1-Score - Chống lỗi đảo dòng):**
   * **PaddleOCR** và **CRAFT + TPS** thể hiện sự vượt trội về **Precision** và **Word F1-Score**. 
   * Nguyên nhân do PaddleOCR (DBNet) và CRAFT định vị Bounding Box / Polygon rất chặt chẽ theo hàng ngang. Kết quả trả về của hai mô hình này rất **thẳng hàng, ngăn nắp và không bị xáo trộn thứ tự đọc**. Khi văn bản chuẩn chỉnh, VietOCR giải mã ít bị rác chữ hơn.
   * **CRAFT + TPS** cho chỉ số **Recall cao nhất (74.80%)** nhờ cơ chế tìm kiếm ở cấp độ ký tự (Character-level affinity), giúp không bỏ sót bất kỳ cụm từ nhỏ lẻ nào trong ảnh.
