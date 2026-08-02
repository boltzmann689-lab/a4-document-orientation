# Phân loại và điều chỉnh góc xoay của tài liệu A4 

Mô hình học Deep Learning có giám sát để phân loại và điều chỉnh góc xoay hướng của tài liệu  ($0^\circ$, $90^\circ$, $180^\circ$, $270^\circ$) sử dụng ResNet18.

---

*Tập trọng số đã huấn luyện: (`best_rotnet_model.pth`):* [Download via Google Drive](https://drive.google.com/file/d/1W6vvsKuBHFEuX6YKELcBGD8KVjwg3076/view?usp=sharing)

## 1.Tổng quan về dự án và phương pháp

* **Tổng quan:** Phân loại ảnh có giám sát (4 classes tương ứng với góc xoay: $0^\circ$, $90^\circ$, $180^\circ$, and $270^\circ$).
* **Kiến trúc hệ thống:** ResNet18 (đã có tham số pretrained trên ImageNet).
* **Dữ liệu:** Sử dụng các ảnh với góc xoay tương ứng để huấn luyện.

---

## 2. Dữ liệu:

Em đã thu thập được **1,000 ảnh gốc** (chưa xoay) với 4 danh mục chính như sau:

| Sub-category | Description / Source | Original Count |
| :--- | :--- | :---: |
| **Scan** | Flatbed scanned documents (DocVQA) | 250 |
| **Receipt** | Receipts (CORD-v2) | 250 |
| **Handwritten** | Full-page handwritten forms (IAM Handwritten Dataset) | 250 |
| **Captured** | Mobile camera captures (SmartDoc) | 250 |
| **Total** | **Diverse multi-source document dataset** | **1,000** |

### Phân chia dữ liệu
Em chia dữ liệu với tỷ lệ **70% Train - 15% Validation - 15% Test**, sau đó để máy xoay ảnh rồi sinh nhãn tự động thu được các tập ảnh tương ứng:
* **Train Set (70%):** 700 original images ($2,800$ rotated samples)
* **Val Set (15%):** 150 original images ($600$ rotated samples)
* **Test Set (15%):** 150 original images ($600$ rotated samples)

---

## 3. Các chỉ số đánh gìá và hiệu suất

| Dataset Split | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Train Set** | 99.82% | 0.9982 | 0.9982 | 0.9982 |
| **Validation Set** | 98.67% | 0.9868 | 0.9867 | 0.9867 |
| **Test Set** | **98.33%** | **0.9835** | **0.9833** | **0.9833** |

### Confusion Matrices
![Confusion Matrices](confusion_matrices_all.png)

### Ý nghĩa các thông số:
* **Accuracy (Độ chính xác tổng thể):**
  Tỷ lệ phần trăm tổng số ảnh được dự đoán đúng góc xoay ($0^\circ, 90^\circ, 180^\circ, 270^\circ$) trên tổng số lượng ảnh được đánh giá.
  $$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Images}}$$

* **Macro Precision (Độ chuẩn xác trung bình):**
  Trung bình cộng độ chuẩn xác của cả 4 lớp góc xoay. Chỉ số này phản ánh **độ tin cậy** khi mô hình đưa ra dự đoán cho một góc xoay cụ thể.
  $$\text{Precision}_i = \frac{\text{TP}_i}{\text{TP}_i + \text{FP}_i} \implies \text{Macro Precision} = \frac{1}{N} \sum_{i=1}^{N} \text{Precision}_i$$
  *(Trong đó: $N = 4$ là số lớp, $\text{TP}_i$ là số mẫu dự đoán đúng của lớp $i$, $\text{FP}_i$ là số mẫu bị đoán nhầm thành lớp $i$).*

* **Macro Recall (Độ nhạy / Độ phủ trung bình):**
  Trung bình cộng độ phủ của cả 4 lớp góc xoay. Chỉ số này đo lường **khả năng phát hiện và không bỏ sót** các ảnh thuộc từng góc xoay thực tế (tỷ lệ bỏ sót thấp).
  $$\text{Recall}_i = \frac{\text{TP}_i}{\text{TP}_i + \text{FN}_i} \implies \text{Macro Recall} = \frac{1}{N} \sum_{i=1}^{N} \text{Recall}_i$$
  *(Trong đó: $\text{FN}_i$ là số mẫu thực tế thuộc lớp $i$ nhưng bị mô hình đoán sót sang lớp khác).*

* **Macro F1-Score (Điểm cân bằng F1 trung bình):**
  Trung bình giữa Macro Precision và Macro Recall. Đây là chỉ số quan trọng nhất đại diện cho hiệu năng tổng thể, đảm bảo mô hình đạt sự cân bằng tốt giữa độ tin cậy (Precision) và khả năng bắt đúng (Recall) mà không bị lệch sang bất kỳ góc xoay nào.
  $$\text{Macro F1-Score} = 2 \times \frac{\text{Macro Precision} \times \text{Macro Recall}}{\text{Macro Precision} + \text{Macro Recall}}$$

### Phân tích lỗi: 
* Mặc dù mô hình chạy tương đối đồng nhất giữa các tập train, val và test, nhưng vẫn có một số trường hợp phân loại sai do ảnh dễ gây nhầm lẫn góc xoay.
---
