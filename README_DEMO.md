# Musical Instrument Recognition Demo

Chương trình demo nhận dạng nhạc cụ sử dụng model đã train trên IRMAS dataset.

## 📋 Yêu cầu

- Python 3.8+
- Các thư viện trong `requirements.txt`

## 🚀 Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

2. Đảm bảo các file model có trong thư mục `IRMAS_Models/`:
   - **CNN Model:**
     - `best_segment_cnn.keras` - Model CNN segment-based
     - `label_encoder_seg.joblib` - Label encoder cho CNN
     - `segment_config.joblib` - Config cho segment-based (tùy chọn)
   - **SVM Model:**
     - `svm_instrument_model.joblib` - SVM pipeline (scaler, selector, model, label_encoder)
     - `label_encoder_svm.joblib` - Label encoder cho SVM (tùy chọn, có thể lấy từ pipeline)

## 💻 Sử dụng

Chạy chương trình:
```bash
python instrument_recognition_demo.py
```

## 🎯 Chức năng

### 1. **Open File**
- Mở file audio (.wav) từ disk
- Hỗ trợ các sample rate khác nhau (tự động resample về 22050 Hz)

### 2. **Record**
- Thu âm từ microphone
- Ghi vào file `recorded_audio.wav`
- Sample rate: 22050 Hz, mono channel

### 3. **Stop**
- Dừng quá trình thu âm

### 4. **Play**
- Phát lại audio đã load/record

### 5. **Prediction Method Selection**
- Chọn phương pháp nhận dạng:
  - **CNN (Segment-based)**: Deep Learning với Mel-Spectrogram và segment aggregation
  - **SVM (Handcrafted)**: Traditional ML với handcrafted features (MFCC, spectral features)

### 6. **Predict**
- Dự đoán nhạc cụ sử dụng phương pháp đã chọn
- Hiển thị:
  - Nhạc cụ dự đoán (top-1)
  - Confidence score
  - Top-3 predictions với confidence
  - Số segments (nếu dùng CNN)

## 🎼 Các nhạc cụ được nhận dạng

1. **Cello** (cel)
2. **Clarinet** (cla)
3. **Flute** (flu)
4. **Acoustic Guitar** (gac)
5. **Electric Guitar** (gel)
6. **Organ** (org)
7. **Piano** (pia)
8. **Saxophone** (sax)
9. **Trumpet** (tru)
10. **Violin** (vio)
11. **Voice** (voi)

## ⚙️ Hai Phương Pháp Nhận Dạng

### 1. **CNN (Segment-based)** - Deep Learning

#### Mel-Spectrogram Parameters
```python
sr = 22050          # Sample rate (Hz)
n_fft = 2048        # FFT window size
hop_length = 512    # Hop length between frames
n_mels = 128        # Number of mel filter banks
```

#### Segment-Based Parameters
```python
segment_duration = 2.0    # Độ dài mỗi segment (giây)
segment_overlap = 0.5     # Overlap ratio (50%) cho sliding window
```

**Input shape của CNN model:** `(128, 87, 1)`
- 128: số mel bins
- 87: số time frames (tương ứng với segment 2.0s)
- 1: channel (grayscale)

#### Cách hoạt động:
1. Audio được cắt thành nhiều segments 2.0s với sliding window (overlap 50%)
2. Mỗi segment được chuyển thành mel-spectrogram
3. CNN predict từng segment
4. **Average softmax** của tất cả segments → kết quả cuối cùng

**Ưu điểm:**
- ✅ Accuracy cao hơn (~85%)
- ✅ Robust với audio dài
- ✅ Tận dụng toàn bộ thông tin trong audio

### 2. **SVM (Handcrafted Features)** - Traditional ML

#### Features được trích xuất:
- **MFCC** (40 coefficients): mean, std, max, min
- **Delta & Delta-Delta** của MFCC
- **Spectral features**: centroid, bandwidth, rolloff, flatness
- **Spectral contrast, Chroma, Tonnetz**
- **Zero crossing rate, RMS**

**Tổng cộng:** 382 features

#### Cách hoạt động:
1. Trích xuất handcrafted features từ toàn bộ audio
2. Normalize và select features (SelectKBest với k=100)
3. SVM predict trực tiếp

**Ưu điểm:**
- ✅ Nhanh hơn CNN
- ✅ Không cần GPU
- ✅ Accuracy ~74%

## 🔧 Điều chỉnh cho model khác

Nếu bạn sử dụng model với cấu trúc khác, cần thay đổi:

### 1. **CNN Parameters** (dòng 44-57):
```python
self.sr = 22050          # Sample rate
self.n_fft = 2048        # FFT window size
self.hop_length = 512    # Hop length
self.n_mels = 128       # Mel filter banks
self.segment_duration = 2.0  # Segment duration (tự động load từ config)
```

### 2. **SVM Parameters** (dòng 48):
```python
self.n_mfcc = 40        # Số MFCC coefficients
```

### 3. **Model paths** (trong hàm `load_models()`):
```python
# CNN
cnn_model_path = "IRMAS_Models/best_segment_cnn.keras"
cnn_label_encoder_path = "IRMAS_Models/label_encoder_seg.joblib"

# SVM
svm_model_path = "IRMAS_Models/svm_instrument_model.joblib"
svm_label_encoder_path = "IRMAS_Models/label_encoder_svm.joblib"
```

### 4. **Label mapping** (dòng 68-81):
```python
self.instrument_names = {
    'cel': 'Cello',
    'cla': 'Clarinet',
    # ... thêm/sửa các nhạc cụ khác
}
```

## 📝 Lưu ý

1. **Audio quality**: Để có kết quả tốt nhất, audio nên:
   - Có độ dài tối thiểu 1-2 giây
   - Chất lượng rõ ràng, ít noise
   - Nhạc cụ phát ra âm thanh rõ ràng

2. **Sample rate**: Audio sẽ tự động được resample về 22050 Hz nếu cần

3. **Real-time**: Hiện tại chương trình xử lý offline. Để chuyển sang real-time:
   - Sử dụng sliding window
   - Xử lý từng chunk audio
   - Cập nhật prediction liên tục

## 🐛 Xử lý lỗi thường gặp

### Lỗi: "Model file not found"
- Kiểm tra đường dẫn đến file model
- Đảm bảo file `best_cnn_model.keras` tồn tại trong `IRMAS_Models/`

### Lỗi: "Failed to load models"
- Kiểm tra version của TensorFlow/Keras
- Đảm bảo model được save đúng format

### Lỗi: "No audio to play"
- Record hoặc mở file audio trước khi play/predict

### Lỗi khi predict: Shape mismatch
- Kiểm tra input shape của model
- Đảm bảo mel-spectrogram parameters khớp với lúc training

## 📚 Cấu trúc code

- `InstrumentRecognitionApp`: Class chính chứa UI và logic
- `load_models()`: Load model và label encoder
- `record_audio()`: Thu âm từ microphone
- `open_file()`: Mở file audio
- `play_audio()`: Phát lại audio
- `extract_mel_spectrogram()`: Trích xuất features
- `prepare_input()`: Chuẩn bị input cho model
- `predict_instrument()`: Dự đoán và hiển thị kết quả

## 🔮 Mở rộng

Có thể mở rộng thêm:
- Hiển thị Mel-Spectrogram visualization
- Real-time streaming prediction
- Export kết quả ra file
- Batch processing nhiều file
- So sánh với ground truth labels

