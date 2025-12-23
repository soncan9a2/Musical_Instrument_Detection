# Quick Start Guide

## 🚀 Chạy chương trình

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Test model (tùy chọn):**
```bash
python test_model.py
```

3. **Chạy chương trình demo:**
```bash
python instrument_recognition_demo.py
```

## 📖 Hướng dẫn sử dụng

### Bước 1: Load audio
- **Option A**: Click "Open File" để chọn file .wav có sẵn
- **Option B**: Click "Record" để thu âm từ microphone, sau đó click "Stop" khi xong

### Bước 2: Nghe lại (tùy chọn)
- Click "Play" để nghe lại audio đã load

### Bước 3: Chọn phương pháp (tùy chọn)
- Chọn **CNN (Segment-based)** hoặc **SVM (Handcrafted)** trong phần "Prediction Method"
- CNN: Chính xác hơn (~85%), nhưng chậm hơn
- SVM: Nhanh hơn, accuracy ~74%

### Bước 4: Nhận dạng
- Click "Predict" để bắt đầu nhận dạng nhạc cụ
- Kết quả sẽ hiển thị:
  - Nhạc cụ dự đoán (top-1)
  - Confidence score (%)
  - Top-3 predictions
  - Số segments (nếu dùng CNN)

## ⚠️ Lưu ý

- Audio nên có độ dài tối thiểu **0.5 giây**
- Chất lượng audio càng tốt, kết quả càng chính xác
- Đảm bảo microphone hoạt động tốt khi thu âm

## 🎼 11 nhạc cụ được nhận dạng

1. Cello (cel)
2. Clarinet (cla)
3. Flute (flu)
4. Acoustic Guitar (gac)
5. Electric Guitar (gel)
6. Organ (org)
7. Piano (pia)
8. Saxophone (sax)
9. Trumpet (tru)
10. Violin (vio)
11. Voice (voi)

## 🔧 Troubleshooting

**Lỗi "Model file not found"**
- **CNN**: Kiểm tra `IRMAS_Models/best_segment_cnn.keras` và `label_encoder_seg.joblib`
- **SVM**: Kiểm tra `IRMAS_Models/svm_instrument_model.joblib` và `label_encoder_svm.joblib`

**Lỗi "Audio quá ngắn"**
- Thu âm hoặc chọn file audio dài hơn 0.5 giây

**Lỗi khi predict**
- Kiểm tra audio có hợp lệ không
- Thử với file audio khác

Xem `README_DEMO.md` để biết thêm chi tiết.

