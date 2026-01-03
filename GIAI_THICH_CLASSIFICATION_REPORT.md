# 📊 GIẢI THÍCH CHI TIẾT CLASSIFICATION REPORT

## 1. `mel_segments = np.array(mel_segments)[..., np.newaxis]`

### Shape trước và sau:

```python
# TRƯỚC:
mel_segments = [segment_to_mel(seg) for seg in segments]
# Mỗi segment_to_mel() → shape: (128, 87)
# Sau np.array() → shape: (N, 128, 87)
# N = số segments (ví dụ: 4)

# SAU:
mel_segments = np.array(mel_segments)[..., np.newaxis]
# Shape: (N, 128, 87, 1)
# Thêm dimension cuối = 1 (channel)
```

### Tại sao cần?

- **CNN yêu cầu:** `(batch, height, width, channels)`
- **Mel-spectrogram:** Ảnh grayscale → cần 1 channel
- **Không có channel:** Model sẽ báo lỗi shape

### Ví dụ:

```python
# 4 segments từ một audio file:
# Trước: (4, 128, 87)  → ❌ Model không nhận được
# Sau:   (4, 128, 87, 1) → ✅ Model nhận được
```

---

## 2. `classification_report` với `target_names`

```python
classification_report(
    y_true_agg,  # Nhãn thực tế
    y_pred_agg,  # Nhãn dự đoán
    target_names=[INSTRUMENT_MAP.get(x, x) for x in label_encoder_seg.classes_]
)
```

### `target_names` làm gì?

- **Đổi tên class** từ code → tên đầy đủ (dễ đọc)
- **INSTRUMENT_MAP:** `{'cel': 'Cello', 'cla': 'Clarinet', ...}`
- **label_encoder_seg.classes_:** `['cel', 'cla', 'flu', ...]`
- **Kết quả:** `['Cello', 'Clarinet', 'Flute', ...]`

### Ví dụ:

```
# KHÔNG có target_names:
cel          0.68      0.91      0.78        78  ← Khó đọc

# CÓ target_names:
Cello        0.68      0.91      0.78        78  ← Dễ đọc!
```

---

## 3. GIẢI THÍCH CÁC THÔNG SỐ

### Ví dụ Classification Report:

```
                 precision    recall  f1-score   support

          Cello       0.68      0.91      0.78        78
       Clarinet       0.79      0.80      0.79       101
          Flute       0.78      0.78      0.78        90
Acoustic Guitar       0.87      0.91      0.89       127
Electric Guitar       0.82      0.85      0.83       152
          Organ       0.85      0.97      0.90       136
          Piano       0.92      0.91      0.92       144
      Saxophone       0.77      0.70      0.73       125
        Trumpet       0.95      0.85      0.90       116
         Violin       0.89      0.61      0.72       116
          Voice       0.94      0.95      0.94       156

       accuracy                           0.85      1341
      macro avg       0.84      0.84      0.84      1341
   weighted avg       0.85      0.85      0.84      1341
```

---

### 📊 CÁC CỘT:

#### 1. **Precision (Độ chính xác dự đoán)**

**Công thức:**
```
Precision = TP / (TP + FP)
```

**Ý nghĩa:**
- Trong số các mẫu model dự đoán là class đó, bao nhiêu % là đúng?
- **Ví dụ Cello (0.68 = 68%):**
  - Model dự đoán 100 mẫu là Cello
  - Trong đó 68 mẫu thực sự là Cello
  - → Precision = 68/100 = 0.68

**Giải thích:**
- **Precision cao** → Model ít dự đoán sai (khi nói "Cello" thì đúng là Cello)
- **Precision thấp** → Model dự đoán sai nhiều (nói "Cello" nhưng không phải)

---

#### 2. **Recall (Độ nhạy)**

**Công thức:**
```
Recall = TP / (TP + FN)
```

**Ý nghĩa:**
- Trong số các mẫu thực tế là class đó, model tìm được bao nhiêu %?
- **Ví dụ Cello (0.91 = 91%):**
  - Có 78 mẫu thực tế là Cello
  - Model tìm được 71 mẫu (91%)
  - → Recall = 71/78 = 0.91

**Giải thích:**
- **Recall cao** → Model ít bỏ sót (tìm được hầu hết các mẫu thực tế)
- **Recall thấp** → Model bỏ sót nhiều (có mẫu là Cello nhưng không tìm được)

---

#### 3. **F1-Score (Trung bình điều hòa)**

**Công thức:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Ý nghĩa:**
- Kết hợp cả Precision và Recall
- **Ví dụ Cello:**
  - Precision = 0.68, Recall = 0.91
  - F1 = 2 × (0.68 × 0.91) / (0.68 + 0.91) = 0.78

**Giải thích:**
- **F1 cao** → Cả Precision và Recall đều tốt
- **F1 thấp** → Một trong hai (hoặc cả hai) thấp

---

#### 4. **Support (Số lượng mẫu)**

**Ý nghĩa:**
- Số lượng mẫu thực tế của class đó trong test set
- **Ví dụ Cello: 78 mẫu**

---

### 📈 PHÂN TÍCH VÍ DỤ (CELLO):

```
Cello: precision=0.68, recall=0.91, f1=0.78, support=78
```

**Giải thích:**
- **Precision 0.68 (68%):**
  - Khi model dự đoán "Cello", 68% là đúng
  - 32% là sai (có thể nhầm với Violin, Organ, ...)
- **Recall 0.91 (91%):**
  - Trong 78 mẫu Cello thực tế, model tìm được 91% (≈71 mẫu)
  - Bỏ sót 9% (≈7 mẫu)
- **F1 0.78:**
  - Cân bằng giữa Precision và Recall
  - Model tìm được nhiều Cello (Recall cao) nhưng đôi khi nhầm (Precision thấp)

**Kết luận:**
- Model **tìm được nhiều Cello** (Recall cao = 91%)
- Nhưng **đôi khi nhầm** (Precision thấp = 68%)
- → Cần cải thiện Precision (giảm false positives)

---

### 📊 CÁC DÒNG TỔNG HỢP:

#### 1. **Accuracy (Độ chính xác tổng thể)**

```
accuracy = 0.85 (85%)
```

**Công thức:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Ý nghĩa:**
- Trong tất cả 1341 mẫu test, model dự đoán đúng 85%
- → 1140 mẫu đúng, 201 mẫu sai

---

#### 2. **Macro Average (Trung bình đơn giản)**

```
macro avg: precision=0.84, recall=0.84, f1=0.84
```

**Công thức:**
```
Macro Precision = (Precision_1 + Precision_2 + ... + Precision_11) / 11
```

**Ý nghĩa:**
- Trung bình đơn giản của tất cả classes
- **Không quan tâm** đến số lượng mẫu của mỗi class
- **Đối xử công bằng** với tất cả classes

**Ví dụ:**
```
Macro Precision = (0.68 + 0.79 + 0.78 + ... + 0.94) / 11 ≈ 0.84
```

---

#### 3. **Weighted Average (Trung bình có trọng số)**

```
weighted avg: precision=0.85, recall=0.85, f1=0.84
```

**Công thức:**
```
Weighted Precision = Σ(Precision_i × Support_i) / Σ(Support_i)
```

**Ý nghĩa:**
- Trung bình có trọng số theo số lượng mẫu
- **Classes có nhiều mẫu** → Trọng số cao hơn
- **Phản ánh tốt hơn** hiệu suất thực tế (vì ưu tiên classes có nhiều data)

**Ví dụ:**
```
Weighted Precision = (0.68×78 + 0.79×101 + ... + 0.94×156) / 1341 ≈ 0.85
```

**Tại sao Weighted > Macro?**
- Classes có nhiều mẫu (Voice: 156, Electric Guitar: 152) có Precision cao
- → Weighted Average cao hơn Macro Average

---

## 🎯 SO SÁNH MACRO vs WEIGHTED:

| Metric | Macro Avg | Weighted Avg |
|--------|-----------|--------------|
| **Cách tính** | Trung bình đơn giản | Trung bình có trọng số |
| **Ưu tiên** | Tất cả classes đều nhau | Classes có nhiều mẫu |
| **Phù hợp** | Class balance | Class imbalance |
| **Kết quả** | 0.84 | 0.85 |

**Trong đồ án này:**
- Dataset có **class imbalance** (Voice: 156, Cello: 78)
- → **Weighted Average** phản ánh tốt hơn hiệu suất thực tế

---

## 📋 TÓM TẮT:

1. **`[..., np.newaxis]`:** Thêm channel dimension (1) → Shape: (N, 128, 87, 1)
2. **`target_names`:** Đổi tên class từ code → tên đầy đủ
3. **Precision:** Độ chính xác khi dự đoán class đó
4. **Recall:** Tỷ lệ tìm được các mẫu thực tế
5. **F1-Score:** Cân bằng giữa Precision và Recall
6. **Support:** Số lượng mẫu thực tế
7. **Macro Avg:** Trung bình đơn giản (đối xử công bằng)
8. **Weighted Avg:** Trung bình có trọng số (ưu tiên classes có nhiều mẫu)

