# HƯỚNG DẪN ĐỒ ÁN NHẬN DẠNG NHẠC CỤ - IRMAS DATASET

## 📋 TỔNG QUAN ĐỒ ÁN

**Tên đồ án:** Nhận dạng nhạc cụ từ tín hiệu âm thanh sử dụng Deep Learning

**Mục tiêu:** Xây dựng mô hình CNN có khả năng nhận dạng 11 loại nhạc cụ từ file audio

**Phương pháp:** Segment-based CNN với Mel-Spectrogram và Focal Loss

**Dataset:** IRMAS (Iowa Recorded Music Assessment) - 11 loại nhạc cụ, ~6,705 file training

---

## 🛠️ CÔNG NGHỆ SỬ DỤNG

### Thư viện chính:

- **TensorFlow/Keras**: Deep Learning framework
- **Librosa**: Xử lý audio và trích xuất Mel-Spectrogram
- **NumPy, Pandas**: Xử lý dữ liệu
- **Scikit-learn**: Chia dữ liệu, đánh giá model
- **Matplotlib, Seaborn**: Visualization

### Môi trường:

- **Google Colab** (khuyến nghị): GPU runtime
- **Python 3.8+**

### Cài đặt dependencies:

**Cho Google Colab:**

```python
!pip install librosa>=0.9.0
!pip install tensorflow>=2.10.0
!pip install scikit-learn>=1.0.0
!pip install numpy>=1.21.0
!pip install pandas
!pip install matplotlib seaborn
!pip install joblib
```

**Cho môi trường local:**

```bash
pip install -r requirements.txt
```

**File `requirements.txt` bao gồm:**

- numpy>=1.21.0
- soundfile>=0.10.0
- sounddevice>=0.4.0
- librosa>=0.9.0
- scikit-learn>=1.0.0
- tensorflow>=2.10.0
- scipy>=1.9.0
- joblib>=1.1.0
- Pillow>=9.0.0

---

## 📊 SƠ LƯỢC CÁC BƯỚC

Đồ án gồm **6 STEP chính**:

1. **STEP 1: DỮ LIỆU (IRMAS DATASET)** - Giới thiệu và cấu trúc dataset
2. **STEP 2: XỬ LÝ DỮ LIỆU** - Chia dữ liệu, trích xuất segments, Mel-Spectrogram
3. **STEP 3: XÂY DỰNG MODEL** - Kiến trúc CNN với Focal Loss
4. **STEP 4: TRAINING MODEL** - Data Augmentation, Callbacks, Training process
5. **STEP 5: ĐÁNH GIÁ MODEL** - Metrics, Testing Data, Confusion Matrix
6. **STEP 6: ỨNG DỤNG THỰC TẾ** - Load model, Real-time Recognition, GUI Application

---

============

## 📁 STEP 1: DỮ LIỆU (IRMAS DATASET)

### 1.1. Giới thiệu IRMAS Dataset

**IRMAS (Iowa Recorded Music Assessment)** là dataset chuẩn cho bài toán nhận dạng nhạc cụ:

- **11 loại nhạc cụ**: Cello, Clarinet, Flute, Acoustic Guitar, Electric Guitar, Organ, Piano, Saxophone, Trumpet, Violin, Voice
- **Training Data**: ~6,705 file audio (mỗi file ~3 giây)
- **Testing Data**: ~2,874 file audio (độ dài khác nhau)

### 1.2. Cấu trúc dữ liệu

#### Training Data:

```
IRMAS-TrainingData/
├── cel/          (388 files)  - Cello
├── cla/          (505 files)  - Clarinet
├── flu/          (451 files)  - Flute
├── gac/          (637 files)  - Acoustic Guitar
├── gel/          (760 files)  - Electric Guitar
├── org/          (682 files)  - Organ
├── pia/          (721 files)  - Piano
├── sax/          (626 files)  - Saxophone
├── tru/          (577 files)  - Trumpet
├── vio/          (580 files)  - Violin
└── voi/          (778 files)  - Voice
```

**Đặc điểm:**

- Mỗi file audio ~3 giây
- Đã được chia sẵn theo từng thư mục nhạc cụ
- Format: `.wav`, sample rate: 22050 Hz
- **Single-label**: Mỗi file chỉ thuộc 1 nhạc cụ

#### Testing Data:

```
IRMAS-TestingData-Part1/
IRMAS-TestingData-Part2/
IRMAS-TestingData-Part3/
├── [file].wav    - File audio
└── [file].txt    - File chứa labels (có thể multi-label)
```

**Đặc điểm:**

- Độ dài file khác nhau (không cố định 3s)
- **Multi-label**: Một file có thể có nhiều nhạc cụ
- Labels được lưu trong file `.txt` cùng tên

### 1.3. Mapping nhạc cụ

| Code | Tên đầy đủ      | Số lượng (Training) |
| ---- | --------------- | ------------------- |
| cel  | Cello           | 388                 |
| cla  | Clarinet        | 505                 |
| flu  | Flute           | 451                 |
| gac  | Acoustic Guitar | 637                 |
| gel  | Electric Guitar | 760                 |
| org  | Organ           | 682                 |
| pia  | Piano           | 721                 |
| sax  | Saxophone       | 626                 |
| tru  | Trumpet         | 577                 |
| vio  | Violin          | 580                 |
| voi  | Voice           | 778                 |

============

## 🔧 STEP 2: XỬ LÝ DỮ LIỆU

### 2.1. Tải và giải nén dataset

**Link tải IRMAS Dataset:**

- IRMAS Training Data: [Link tải TrainingData]
- IRMAS Testing Data: [Link tải TestingData-Part1, Part2, Part3]

**Lưu ý:** Dataset có thể tải từ trang chủ IRMAS hoặc các nguồn academic khác.

**Cấu trúc thư mục sau khi giải nén:**

```
IRMAS/
├── TrainingData/
│   ├── [cel]/
│   ├── [cla]/
│   ├── [flu]/
│   ├── [gac]/
│   ├── [gel]/
│   ├── [org]/
│   ├── [pia]/
│   ├── [sax]/
│   ├── [tru]/
│   ├── [vio]/
│   └── [voi]/
└── TestingData/
    ├── Part1/
    ├── Part2/
    └── Part3/
```

### 2.2. Giải nén dataset

```python
# Giải nén các file zip
extract_zip('IRMAS-TrainingData.zip', WORK_DIR)
extract_zip('IRMAS-TestingData-Part1.zip', WORK_DIR)
extract_zip('IRMAS-TestingData-Part2.zip', WORK_DIR)
extract_zip('IRMAS-TestingData-Part3.zip', WORK_DIR)
```

### 2.2. Load và chia dữ liệu

**Quan trọng:** Chia dữ liệu ở cấp độ **FILE** trước khi cắt segments (tránh data leakage)

**⚠️ LƯU Ý QUAN TRỌNG: Có 2 loại Test khác nhau!**

1. **Test từ TrainingData** (chia từ 6,705 files):

   - Dùng để đánh giá model trong quá trình phát triển
   - Đảm bảo model không overfit

2. **TestingData riêng** (2,874 files):
   - Dataset riêng biệt, **KHÔNG nằm trong TrainingData**
   - Dùng để đánh giá cuối cùng trên data hoàn toàn mới
   - Đây là "real-world" test

**Quy trình chia TrainingData (6,705 files) thành 2 bước:**

**Bước 1:** Chia Training Data thành Train+Val (80%) và Test (20%)

```python
# Bước 1: Chia 80% train+val và 20% test
train_files, test_files, train_labels, test_labels = train_test_split(
    file_paths, labels,
    test_size=0.2,        # 20% cho test (từ TrainingData)
    random_state=42,
    stratify=labels      # Giữ tỷ lệ các class
)
```

**Kết quả:**

- **Train + Val**: 80% = 5,364 files
- **Test (từ TrainingData)**: 20% = 1,341 files

**Bước 2:** Chia Train+Val thành Train (85%) và Val (15%)

```python
# Bước 2: Chia train+val thành train (85%) và val (15%)
train_files, val_files, train_labels, val_labels = train_test_split(
    train_files, train_labels,
    test_size=0.15,       # 15% cho val
    random_state=42,
    stratify=train_labels
)
```

**Kết quả cuối cùng từ TrainingData:**

- **Train**: 85% × 80% = **68%** (~4,559 files)
- **Val**: 15% × 80% = **12%** (~805 files)
- **Test (từ TrainingData)**: **20%** (~1,341 files)

**Tổng: 100% (6,705 files từ TrainingData)**

**TestingData riêng (2,874 files):**

- **KHÔNG** được dùng trong quá trình training
- Chỉ dùng để đánh giá cuối cùng sau khi model đã train xong
- Đây là dataset "unseen" - model chưa từng thấy

**⚠️ QUAN TRỌNG: Khi nào dùng từng set?**

**Trong quá trình Training (mỗi epoch):**

```python
history = model.fit(
    train_gen_seg,                    # Train set → Dùng để train
    validation_data=(X_val_seg, ...),  # Val set → Dùng MỖI EPOCH để monitor
    epochs=100,
    callbacks=[...]                    # EarlyStopping dựa trên val_accuracy
)
```

- **Train set**: Dùng để train model (mỗi epoch)
- **Val set**: Dùng MỖI EPOCH để:
  - Monitor `val_accuracy` và `val_loss`
  - Chọn best model (ModelCheckpoint)
  - EarlyStopping nếu không cải thiện
  - ReduceLROnPlateau nếu loss không giảm

**SAU KHI Training xong:**

- **Test set (từ TrainingData)**: Dùng để đánh giá model sau khi train xong

  - Không dùng trong quá trình training
  - Chỉ dùng 1 lần sau khi training hoàn tất
  - Để kiểm tra model có overfit không

- **TestingData riêng**: Dùng để đánh giá cuối cùng trên data hoàn toàn mới

**Tóm tắt:**

| Set                        | Khi nào dùng               | Mục đích                                 |
| -------------------------- | -------------------------- | ---------------------------------------- |
| **Train**                  | Mỗi epoch                  | Train model                              |
| **Val**                    | Mỗi epoch                  | Monitor, chọn best model, early stopping |
| **Test (từ TrainingData)** | Sau khi train xong (1 lần) | Đánh giá model, kiểm tra overfit         |
| **TestingData riêng**      | Sau khi train xong (1 lần) | Đánh giá cuối cùng trên data mới         |

**Tại sao chia 2 bước?**

- Đảm bảo test set độc lập hoàn toàn
- Val set dùng để chọn best model trong quá trình training (mỗi epoch)
- Test set (từ TrainingData) dùng để đánh giá SAU KHI training xong (không dùng mỗi epoch)
- TestingData riêng dùng để đánh giá cuối cùng trên data hoàn toàn mới

### 2.3. Segment-based Strategy

**Tại sao dùng Segment-based?**

- Dataset nhỏ (~6,705 files) → Cần tăng data
- Audio 3s → Cắt thành nhiều segments → Tăng data ×6 lần
- Segment 2.0s → Capture tốt hơn nhạc cụ sustain (organ, violin)
- Prediction robust hơn nhờ **weighted average aggregation** (segment có confidence cao được ưu tiên)

**Cấu hình:**

```python
SEGMENT_DURATION = 2.0      # Độ dài mỗi segment (giây)
NUM_SEGMENTS_TRAIN = 6      # Số segments random cho mỗi file khi train
SEGMENT_OVERLAP = 0.5       # Overlap 50% cho sliding window khi test
```

**Aggregation Strategy:**

- **Training**: Random segments → Tăng diversity, tăng data
- **Validation/Testing**: Sliding window → **Weighted Average** (segment có confidence cao hơn được ưu tiên)

**Cách hoạt động:**

#### Training: Random Segments

**Từ file audio 3s → Cắt 6 segments ngẫu nhiên, mỗi segment 2.0s**

```
File audio: [==========] (3 giây)
            |--2s--|  |--2s--|  |--2s--|  ... (6 segments random)
```

**Ví dụ:**

- File audio 3s (66,150 samples với sr=22050)
- Cắt 6 segments, mỗi segment 2.0s (44,100 samples)
- Các segments có thể **overlap** (chồng lên nhau) hoặc **không overlap**
- Vị trí bắt đầu của mỗi segment là **ngẫu nhiên**

```python
def extract_random_segments(y, num_segments=6):
    """
    Cắt 6 segments ngẫu nhiên từ audio 3s
    Mỗi segment dài 2.0s
    """
    segments = []
    for _ in range(6):
        start = np.random.randint(0, max_start)  # Vị trí ngẫu nhiên
        segment = y[start:start + 44100]  # 2.0s = 44100 samples
        segments.append(segment)
    return segments
```

**Kết quả:** 1 file → 6 segments → Data tăng ×6

#### Validation/Testing: Sliding Window

**Từ file audio → Cắt segments với sliding window (overlap 50%)**

```
File audio: [==========] (3 giây)
            |--2s--|
                 |--2s--|  (overlap 50%)
                      |--2s--|
```

**Ví dụ với file 3s:**

- Segment 1: 0.0s → 2.0s
- Segment 2: 1.0s → 3.0s (overlap 50% với segment 1)
- **Kết quả:** 2 segments từ 1 file

**Ví dụ với file 5s:**

- Segment 1: 0.0s → 2.0s
- Segment 2: 1.0s → 3.0s
- Segment 3: 2.0s → 4.0s
- Segment 4: 3.0s → 5.0s
- **Kết quả:** 4 segments từ 1 file

**Công thức tính số segments với sliding window:**

```
hop = segment_samples × (1 - overlap)
số_segments = ⌊(audio_length - segment_samples) / hop⌋ + 1
```

Trong đó:

- `hop`: Bước nhảy giữa các segments
- `overlap`: Tỷ lệ overlap (0.5 = 50%)
- `segment_samples`: Số samples trong 1 segment
- `audio_length`: Độ dài audio (samples)

**Ví dụ với audio 5s (110,250 samples) và segment 2s (44,100 samples), overlap 50%:**

- `hop = 44,100 × (1 - 0.5) = 22,050 samples`
- `số_segments = ⌊(110,250 - 44,100) / 22,050⌋ + 1 = ⌊3⌋ + 1 = 4 segments`

```python
def extract_sliding_segments(y, overlap=0.5):
    """
    Cắt segments với sliding window
    Overlap 50% → Mỗi segment mới bắt đầu từ giữa segment trước
    """
    segments = []
    step = int(segment_samples * (1 - overlap))  # Bước nhảy = 50% segment
    for start in range(0, len(y) - segment_samples, step):
        segment = y[start:start + segment_samples]
        segments.append(segment)
    return segments
```

**Kết quả:**

- **Train**: 4,559 files → **27,354 segments** (×6)
- **Val**: 805 files → **1,610 segments** (×2 trung bình)
- **Test**: 1,341 files → **2,682 segments** (×2 trung bình)

**Tại sao khác nhau?**

- **Training**: Random 6 segments → Tăng diversity, tăng data
- **Val/Test**: Sliding window → Cover toàn bộ audio, đảm bảo consistency
- **Aggregation**: Dùng **Weighted Average** (không phải simple average) → Segment có confidence cao được ưu tiên

### 2.4. Trích xuất Mel-Spectrogram

**Mel-Spectrogram** là biểu diễn tần số của âm thanh theo thời gian:

- Chuyển đổi audio signal → 2D image (frequency × time)
- Model CNN xử lý như ảnh grayscale

#### 2.4.1. Các khái niệm cơ bản

**Audio Signal (1D):**

```
Audio: [sample1, sample2, sample3, ..., sampleN]
       ↑
       Tín hiệu số hóa theo thời gian
```

**Spectrogram (2D):**

```
Frequency (Hz)
    ↑
    |  [intensity]
    |  [intensity]
    |  [intensity]
    |  ...
    └────────────────────────→ Time
```

#### 2.4.2. Các tham số quan trọng

**Tham số:**

```python
SAMPLE_RATE = 22050        # Hz - Tần số lấy mẫu
N_MELS = 128               # Số mel filter banks (mel bins)
N_FFT = 2048               # FFT window size
HOP_LENGTH = 512           # Hop length giữa các frames
```

**Giải thích từng tham số:**

##### 1. **SAMPLE_RATE (22050 Hz)**

- Số samples lấy trong 1 giây
- 22050 Hz = 22,050 samples/giây
- Segment 2.0s = 44,100 samples

##### 2. **N_FFT (2048) - FFT Window Size**

- Kích thước cửa sổ FFT (Fast Fourier Transform)
- Dùng để chuyển đổi từ time domain → frequency domain
- **Càng lớn → độ phân giải tần số càng cao, nhưng độ phân giải thời gian thấp hơn**

**Ví dụ:**

```
N_FFT = 2048 → Phân tích 2048 samples mỗi lần
→ Tần số tối đa có thể phân tích = SAMPLE_RATE / 2 = 11025 Hz
```

##### 3. **HOP_LENGTH (512) - Bước nhảy giữa các frames**

- **Quan trọng nhất để hiểu tại sao có 87 time frames!**

**Hop Length là gì?**

- Khoảng cách (số samples) giữa 2 cửa sổ FFT liên tiếp
- Hop length = 512 → Mỗi frame cách nhau 512 samples

**Ví dụ:**

```
Frame 1: samples [0:2048]      (N_FFT = 2048)
Frame 2: samples [512:2560]    (nhảy 512 samples = HOP_LENGTH)
Frame 3: samples [1024:3072]   (nhảy thêm 512 samples)
Frame 4: samples [1536:3584]   (nhảy thêm 512 samples)
...
```

**Tại sao dùng Hop Length?**

- Nếu không có hop (hop = N_FFT): Mất thông tin, không overlap
- Có hop < N_FFT: Có overlap → Capture tốt hơn các thay đổi theo thời gian

##### 4. **N_MELS (128) - Số Mel Bins**

**Mel Bins là gì?**

- **Mel scale**: Thang đo tần số phù hợp với cách tai người cảm nhận
- Tai người nhạy cảm hơn với tần số thấp (200-2000 Hz) so với tần số cao
- Mel scale: Chia tần số thành các "bins" (thùng) theo thang Mel

**Công thức chuyển đổi Hz → Mel:**

```
mel(f) = 2595 · log₁₀(1 + f/700)
```

Trong đó:

- `f`: Tần số (Hz)
- `mel(f)`: Tần số theo thang Mel

**Ví dụ:**

```
Tần số tuyến tính: 0Hz, 100Hz, 200Hz, 300Hz, ... (cách đều)
Mel scale:        0,   100,   200,   300,   ... (không cách đều, tập trung ở tần số thấp)
```

**N_MELS = 128:**

- Chia tần số thành 128 bins theo thang Mel
- Mỗi bin đại diện cho một dải tần số
- **128 mel bins = 128 hàng trong mel-spectrogram**

#### 2.4.3. Tính toán số Time Frames (87)

**Công thức tính số time frames:**

```
Số time frames = (Số samples - N_FFT) / HOP_LENGTH + 1
```

**Với segment 2.0s:**

```python
SEGMENT_DURATION = 2.0      # giây
SAMPLE_RATE = 22050         # Hz
SEGMENT_SAMPLES = 2.0 × 22050 = 44,100 samples

N_FFT = 2048
HOP_LENGTH = 512

Số time frames = (44,100 - 2048) / 512 + 1
                = 42,052 / 512 + 1
                = 82.13 + 1
                = 83.13
                ≈ 87 frames (do làm tròn và padding)
```

**Giải thích:**

- Segment có 44,100 samples
- Frame đầu tiên: samples [0:2048]
- Frame cuối cùng: samples [42,052:44,100]
- Số frames = (44,100 - 2048) / 512 + 1 ≈ 87

**Tại sao +1?**

- Frame đầu tiên bắt đầu ở sample 0
- Mỗi frame tiếp theo nhảy 512 samples
- Cần +1 để tính frame đầu tiên

#### 2.4.4. Input Shape: (128, 87, 1)

**Input shape:** `(128, 87, 1)`

- **128**: Số mel bins (frequency axis - chiều cao)
- **87**: Số time frames (time axis - chiều rộng)
- **1**: Channel (grayscale - giống ảnh đen trắng)

**Hình dung:**

```
Mel-Spectrogram (128 × 87):

Frequency (128 bins)
    ↑
128 | [intensity] [intensity] ... [intensity]  ← Frame 1
127 | [intensity] [intensity] ... [intensity]  ← Frame 2
... |    ...         ...      ...    ...
  1 | [intensity] [intensity] ... [intensity]
  0 | [intensity] [intensity] ... [intensity]  ← Frame 87
    └─────────────────────────────────────────→ Time
       0s    0.023s  0.046s  ...   2.0s
       (mỗi frame cách nhau 512/22050 ≈ 0.023s)
```

#### 2.4.5. Công thức trích xuất

```python
def segment_to_mel(segment):
    """
    Chuyển đổi audio segment → Mel-Spectrogram

    Input: segment (44,100 samples = 2.0s)
    Output: mel_spec_db (128 × 87)
    """
    # 1. Tính Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=segment,          # Audio signal (44,100 samples)
        sr=22050,           # Sample rate
        n_mels=128,         # Số mel bins
        n_fft=2048,         # FFT window size
        hop_length=512      # Hop length giữa các frames
    )
    # Output: mel_spec shape = (128, 87)

    # 2. Chuyển sang decibel (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Giá trị từ -∞ đến 0 dB (normalize về [0, 1] sau)

    return mel_spec_db
```

#### 2.4.6. Tóm tắt

| Tham số              | Giá trị  | Ý nghĩa                   |
| -------------------- | -------- | ------------------------- |
| **SAMPLE_RATE**      | 22050 Hz | Số samples/giây           |
| **N_FFT**            | 2048     | Kích thước cửa sổ FFT     |
| **HOP_LENGTH**       | 512      | Bước nhảy giữa các frames |
| **N_MELS**           | 128      | Số mel bins (frequency)   |
| **SEGMENT_DURATION** | 2.0s     | Độ dài segment            |
| **SEGMENT_SAMPLES**  | 44,100   | Số samples trong segment  |
| **Time Frames**      | 87       | Số frames theo thời gian  |

**Công thức quan trọng:**

```
Time Frames = (SEGMENT_SAMPLES - N_FFT) / HOP_LENGTH + 1
            = (44,100 - 2048) / 512 + 1
            ≈ 87
```

============

## 🏗️ STEP 3: XÂY DỰNG MODEL

### 3.1. Kiến trúc Model

**CNN với các Conv Blocks:**

```
Input (128, 87, 1)
    ↓
Block 1: Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm
    ↓ MaxPooling2D + Dropout(0.25)
Block 2: Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm
    ↓ MaxPooling2D + Dropout(0.25)
Block 3: Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm
    ↓ MaxPooling2D + Dropout(0.3)
Block 4: Conv2D(256) → BatchNorm → Conv2D(256) → BatchNorm
    ↓ MaxPooling2D + Dropout(0.3)
Global Average Pooling
    ↓
Dense (512) → BatchNorm → Dropout(0.5)
    ↓
Dense (256) → BatchNorm → Dropout(0.5)
    ↓
Output (11 classes) - Softmax
```

**Công thức Softmax:**

```
p_i = exp(z_i) / Σⱼ exp(z_j)
```

Trong đó:

- `z_i`: Logit (raw output) của class i
- `p_i`: Probability của class i sau softmax
- `Σⱼ p_j = 1` (tổng probabilities = 1)

### 3.2. Loss Function: Focal Loss

**Công thức gốc:**

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

Trong đó:

- `p_t`: Probability của true class
  - `p_t = p` nếu y = 1
  - `p_t = 1 - p` nếu y = 0
- `α_t`: Class weight (trong code: không dùng, α = 1)
- `γ`: Focusing parameter (gamma = 2.0)

**Công thức Cross Entropy (CE) - cơ sở của Focal Loss:**

```
CE = -Σ y_i · log(p_i)
```

Trong đó:

- `y_i`: True label (one-hot encoding)
- `p_i`: Predicted probability cho class i
- Với multi-class: `CE = -log(p_t)` với `p_t` là probability của true class

**Focal Loss = Focal Weight × Cross Entropy:**

```
FL = (1 - p_t)^γ · CE
   = (1 - p_t)^γ · (-log(p_t))
```

**Giải thích:**

- `(1 - p_t)^γ`: Focal weight
  - `p_t` cao (dễ predict) → `(1 - p_t)^γ` nhỏ → Weight thấp
  - `p_t` thấp (khó predict) → `(1 - p_t)^γ` lớn → Weight cao
- Tập trung vào các sample khó phân biệt

**Focal Loss** tập trung vào các sample khó phân biệt:

- `gamma=2.0`: Tập trung vừa phải vào hard examples
- Giúp cải thiện precision cho các class yếu (Saxophone, Trumpet)

**Trong code:**

```python
def focal_loss(gamma=2.0):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)

        # Cross entropy
        ce = -y_true * keras.backend.log(y_pred)

        # p_t: probability của true class
        p_t = keras.backend.sum(y_true * y_pred, axis=-1, keepdims=True)

        # Focal weight: (1 - p_t)^γ
        focal_weight = keras.backend.pow((1 - p_t), gamma)

        # Focal Loss
        loss = focal_weight * ce
        return keras.backend.mean(loss)

    return focal_loss_fixed
```

**Ví dụ:**

```
Sample dễ (p_t = 0.9):
  FL = (1 - 0.9)² × CE = 0.01 × CE → Weight rất thấp

Sample khó (p_t = 0.3):
  FL = (1 - 0.3)² × CE = 0.49 × CE → Weight cao hơn 49 lần
```

### 3.5. Regularization

**L2 Regularization (Weight Decay):**

**Công thức:**

```
L_total = L_loss + λ · Σ θ²
```

Trong đó:

- `L_loss`: Loss function (Focal Loss)
- `λ`: Regularization coefficient (0.001)
- `Σ θ²`: Sum of squared weights

**Gradient với L2:**

```
∂L/∂θ = ∂L_loss/∂θ + 2λ · θ
```

**Batch Normalization:**

**Công thức:**

```
μ_B = (1/m) · Σᵢ₌₁ᵐ x_i                    # Mean của batch
σ²_B = (1/m) · Σᵢ₌₁ᵐ (x_i - μ_B)²         # Variance của batch
x̂_i = (x_i - μ_B) / √(σ²_B + ε)            # Normalize
y_i = γ · x̂_i + β                           # Scale and shift
```

**Công thức đầy đủ:**

1. **Mean:** `μ_B = (1/m) · Σᵢ₌₁ᵐ x_i`
2. **Variance:** `σ²_B = (1/m) · Σᵢ₌₁ᵐ (x_i - μ_B)²`
3. **Normalize:** `x̂_i = (x_i - μ_B) / √(σ²_B + ε)`
4. **Scale and Shift:** `y_i = γ · x̂_i + β`

Trong đó:

- `m`: Batch size
- `ε = 1e-5`: Epsilon (tránh chia 0)
- `γ`: Scale parameter (learnable)
- `β`: Shift parameter (learnable)

**Dropout:**

**Công thức:**

```
y_i = {
    x_i / (1 - p)  với xác suất (1 - p)  # Giữ lại
    0              với xác suất p         # Drop
}
```

Trong đó:

- `p`: Dropout rate (0.25, 0.3, 0.5)
- Training: Áp dụng dropout
- Inference: Không dropout (hoặc scale bởi 1-p)

**Tóm tắt:**

- **L2 Regularization** (weight_decay=0.01): Giảm overfitting
- **BatchNormalization**: Ổn định training
- # **Dropout** (0.25-0.5): Giảm overfitting

## 🎯 STEP 4: TRAINING MODEL

### 4.1. Data Augmentation

**⚠️ PHÂN BIỆT: Segment-based vs Data Augmentation**

**Segment-based (đã giải thích ở Step 2.3):**

- **Cắt audio signal** → Tạo nhiều segments từ 1 file audio
- **Input**: Audio signal (1D array)
- **Output**: Nhiều segments (1 file → 6 segments)
- **Mục đích**: Tăng số lượng data (1 file → 8 samples)

**Data Augmentation (phần này):**

- **Biến đổi mel-spectrogram** → Tạo biến thể từ 1 segment đã có
- **Input**: Mel-spectrogram (128 × 87) đã có
- **Output**: Mel-spectrogram đã được biến đổi
- **Mục đích**: Tăng diversity, giảm overfitting

**Quy trình:**

```
1. Audio file (3s)
   ↓
2. Segment-based: Cắt thành 6 segments (mỗi segment 2.0s)
   ↓
3. Mỗi segment → Mel-spectrogram (128 × 87)
   ↓
4. Data Augmentation: Biến đổi mel-spectrogram (SpecAugment, Mixup)
   ↓
5. Model training
```

**Tại sao cần Data Augmentation?**

- Dataset nhỏ (~6,705 files) → Cần tăng diversity
- Giảm overfitting → Model generalize tốt hơn
- Tăng robustness → Model chịu được noise, biến đổi

**2 kỹ thuật được sử dụng:**

1. **SpecAugment**: Che một phần mel-spectrogram
2. **Mixup**: Trộn 2 samples với nhau

#### 4.1.1. SpecAugment (Spectral Augmentation)

**SpecAugment** là kỹ thuật augmentation cho spectrogram, tương tự như Cutout cho ảnh.

**Cách hoạt động:**

**1. Time Masking** - Che theo trục thời gian:

```
Mel-Spectrogram gốc (128 × 87):
┌─────────────────────────────────┐
│ ████████████████████████████████ │ ← Frequency bin 127
│ ████████████████████████████████ │
│ ████████████████████████████████ │
│ ████████████████████████████████ │
│ ████████████████████████████████ │
│ ...                             │
│ ████████████████████████████████ │ ← Frequency bin 0
└─────────────────────────────────┘
    0s    0.5s   1.0s   1.5s   2.0s
         ↑ Time axis (87 frames)

Sau Time Masking (che 10 frames từ frame 20):
┌─────────────────────────────────┐
│ ████████░░░░░░░░░░██████████████ │ ← Che theo thời gian
│ ████████░░░░░░░░░░██████████████ │
│ ████████░░░░░░░░░░██████████████ │
│ ████████░░░░░░░░░░██████████████ │
│ ████████░░░░░░░░░░██████████████ │
│ ...                             │
│ ████████░░░░░░░░░░██████████████ │
└─────────────────────────────────┘
    0s    0.5s   1.0s   1.5s   2.0s
         ↑ Che từ frame 20-30
```

**2. Frequency Masking** - Che theo trục tần số:

```
Mel-Spectrogram gốc (128 × 87):
┌─────────────────────────────────┐
│ ████████████████████████████████ │ ← Frequency bin 127
│ ████████████████████████████████ │
│ ████████████████████████████████ │
│ ████████████████████████████████ │ ← Che 15 bins từ bin 50
│ ░░░░░░░░░░░░░░░░████████████████ │ ← Frequency bin 50-65
│ ░░░░░░░░░░░░░░░░████████████████ │
│ ░░░░░░░░░░░░░░░░████████████████ │
│ ...                             │
│ ████████████████████████████████ │ ← Frequency bin 0
└─────────────────────────────────┘
    0s    0.5s   1.0s   1.5s   2.0s
```

**Công thức:**

**Time Masking:**

```
x[t₀ : t₀ + t, :] = 0
```

- `t`: Random từ [0, time_mask_param] (25 frames)
- `t₀`: Random start position

**Frequency Masking:**

```
x[:, f₀ : f₀ + f] = 0
```

- `f`: Random từ [0, freq_mask_param] (20 bins)
- `f₀`: Random start position

**Tham số trong code:**

```python
def _spec_augment(self, batch,
                  time_mask_param=25,    # Che tối đa 25 frames
                  freq_mask_param=20,    # Che tối đa 20 frequency bins
                  num_masks=2):          # Áp dụng 2 lần masking
    """
    SpecAugment: Time và Frequency Masking
    Công thức: x[t₀:t₀+t, :] = 0 (time), x[:, f₀:f₀+f] = 0 (frequency)
    """
    for i in range(len(batch)):
        for _ in range(num_masks):  # Áp dụng 2 lần
            # Time masking: Che ngẫu nhiên 0-25 frames
            t = np.random.randint(0, time_mask_param)
            t0 = np.random.randint(0, max(1, time_steps - t))
            batch[i, t0:t0+t, :, :] = 0  # Set = 0 (che)

            # Frequency masking: Che ngẫu nhiên 0-20 bins
            f = np.random.randint(0, freq_mask_param)
            f0 = np.random.randint(0, max(1, freq_bins - f))
            batch[i, :, f0:f0+f, :] = 0  # Set = 0 (che)

    return batch
```

**Lợi ích:**

- Model học được các đặc trưng quan trọng, không phụ thuộc vào một phần cụ thể
- Robust với noise, mất tín hiệu tạm thời
- Giảm overfitting

**Ví dụ thực tế:**

- Time masking: Mô phỏng tín hiệu bị mất trong thời gian ngắn
- Frequency masking: Mô phỏng một số tần số bị nhiễu

#### 4.1.2. Mixup Augmentation

**Mixup** là kỹ thuật trộn 2 samples với nhau để tạo sample mới.

**Cách hoạt động:**

**Công thức:**

```
x_mixed = λ × x1 + (1 - λ) × x2
y_mixed = λ × y1 + (1 - λ) × y2
```

Trong đó:

- `λ` (lambda) được lấy từ phân phối Beta(α, α)
- `α = 0.2` (tham số trong code)
- `x1, x2`: 2 mel-spectrograms
- `y1, y2`: 2 labels (one-hot encoding)

**Công thức gốc:**

```
x_mixed = λ · x₁ + (1 - λ) · x₂
y_mixed = λ · y₁ + (1 - λ) · y₂
```

Trong đó:

- `λ ~ Beta(α, α)`: Mixing coefficient
- `α = 0.2`: Tham số Beta distribution
- `x₁, x₂`: 2 samples ngẫu nhiên
- `y₁, y₂`: 2 labels tương ứng

**Beta Distribution:**

```
Beta(α, α) với α = 0.2
```

- `α` nhỏ → `λ` thường gần 0 hoặc 1 (ít khi ở giữa)
- `α` lớn → `λ` thường ở giữa (0.5)

**Ví dụ:**

```
Sample 1: Piano (label = [0,0,0,0,0,0,1,0,0,0,0])
Sample 2: Guitar (label = [0,0,0,1,0,0,0,0,0,0,0])

λ = 0.3 (ngẫu nhiên từ Beta(0.2, 0.2))

x_mixed = 0.3 × Piano_mel + 0.7 × Guitar_mel
y_mixed = 0.3 × [0,0,0,0,0,0,1,0,0,0,0] + 0.7 × [0,0,0,1,0,0,0,0,0,0,0]
        = [0, 0, 0, 0.7, 0, 0, 0.3, 0, 0, 0, 0]
        = 70% Guitar + 30% Piano
```

**Hình dung:**

```
Mel-Spectrogram 1 (Piano):        Mel-Spectrogram 2 (Guitar):
┌─────────────────┐              ┌─────────────────┐
│ ███████████████ │              │ ░░░░░░░░░░░░░░░ │
│ ███████████████ │              │ ░░░░░░░░░░░░░░░ │
│ ███████████████ │              │ ░░░░░░░░░░░░░░░ │
│ ...             │              │ ...             │
└─────────────────┘              └─────────────────┘

Mixup (λ = 0.3):
┌─────────────────┐
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ← 30% Piano + 70% Guitar
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
│ ...             │
└─────────────────┘
```

**Code:**

```python
def _mixup(self, x, y, alpha=0.2):
    """
    Mixup augmentation
    alpha: Tham số Beta distribution (alpha càng nhỏ → λ gần 0 hoặc 1)
    """
    batch_size = len(x)
    indices = np.random.permutation(batch_size)  # Xáo trộn indices

    # Lấy λ từ Beta distribution
    lam = np.random.beta(alpha, alpha)
    # alpha=0.2 → λ thường gần 0 hoặc 1 (ít khi ở giữa)

    # Trộn 2 samples
    x_mixed = lam * x + (1 - lam) * x[indices]
    y_mixed = lam * y + (1 - lam) * y[indices]

    return x_mixed, y_mixed
```

**Lợi ích:**

- Tăng diversity: Tạo ra các sample mới không có trong dataset
- Model học được ranh giới giữa các class tốt hơn
- Giảm overfitting

**Lưu ý:**

- Chỉ áp dụng cho **training** (không áp dụng cho validation/test)
- Áp dụng với xác suất 50% (trong code: `if np.random.random() > 0.5`)

#### 4.1.3. Data Generator

**SegmentDataGenerator** kết hợp cả 2 kỹ thuật:

```python
class SegmentDataGenerator(keras.utils.Sequence):
    def __init__(self, x, y, batch_size=64,
                 augment=True,    # Bật SpecAugment
                 mixup=True):      # Bật Mixup
        self.x = x
        self.y = y
        self.augment = augment
        self.mixup = mixup

    def __getitem__(self, idx):
        # Lấy batch
        x_batch = self.x[batch_indices].copy()
        y_batch = self.y[batch_indices].copy()

        # 1. Áp dụng SpecAugment (nếu bật)
        if self.augment:
            x_batch = self._spec_augment(x_batch)

        # 2. Áp dụng Mixup (nếu bật, với xác suất 50%)
        if self.mixup and np.random.random() > 0.5:
            x_batch, y_batch = self._mixup(x_batch, y_batch)

        return x_batch, y_batch
```

**Sử dụng:**

```python
# Training: Có augmentation
train_gen = SegmentDataGenerator(
    X_train_seg, y_train_cat_seg,
    batch_size=64,
    augment=True,      # SpecAugment
    mixup=True         # Mixup
)

# Validation: KHÔNG có augmentation
val_gen = SegmentDataGenerator(
    X_val_seg, y_val_cat_seg,
    batch_size=64,
    augment=False,     # Không augmentation
    mixup=False,       # Không mixup
    shuffle=False      # Không shuffle
)
```

**Tại sao validation không có augmentation?**

- Validation dùng để đánh giá model trên data "thật"
- Augmentation chỉ dùng để train model tốt hơn
- Đảm bảo đánh giá chính xác

#### 4.1.4. Tóm tắt

| Kỹ thuật        | Cách hoạt động                                | Lợi ích                               | Áp dụng cho             |
| --------------- | --------------------------------------------- | ------------------------------------- | ----------------------- |
| **SpecAugment** | Che một phần mel-spectrogram (time/frequency) | Robust với noise, giảm overfitting    | Training                |
| **Mixup**       | Trộn 2 samples với nhau                       | Tăng diversity, học ranh giới tốt hơn | Training (50% xác suất) |

**Kết quả:**

- Tăng diversity của training data
- Model generalize tốt hơn
- Giảm overfitting
- Tăng accuracy trên test set

### 4.2. Callbacks

**EarlyStopping:**

- Monitor: `val_accuracy`
- Patience: 15 epochs
- Restore best weights

**ReduceLROnPlateau:**

- Monitor: `val_loss`
- Factor: 0.5 (giảm 50%)
- Patience: 7 epochs
- Min LR: 1e-7

**ModelCheckpoint:**

- Save best model dựa trên `val_accuracy`
- Path: `/content/best_segment_cnn.keras`

### 4.3. Training Configuration

```python
EPOCHS_SEG = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
OPTIMIZER = AdamW(learning_rate=0.001, weight_decay=0.01)
LOSS = Focal Loss (gamma=2.0)
CLASS_WEIGHT = 'balanced'  # Tính class weights để xử lý class imbalance
```

**Class Weights:**

- Tính toán class weights để xử lý class imbalance trong dataset
- Dùng `compute_class_weight('balanced', ...)` từ scikit-learn
- Class có ít samples sẽ có weight cao hơn

```python
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
```

**Training process:**

1. Load data generators với augmentation
2. Compile model với Focal Loss
3. Fit với callbacks và class_weight
4. Model tự động save best weights

**Thời gian training:** ~2-3 giờ trên GPU Colab (100 epochs)

============

## 📈 STEP 5: ĐÁNH GIÁ MODEL

### 5.1. Metrics

**Accuracy:** Tỷ lệ dự đoán đúng
**Precision:** Tỷ lệ dự đoán đúng trong số các dự đoán
**Recall:** Tỷ lệ tìm được trong số các mẫu thực tế
**F1-Score:** Trung bình điều hòa của Precision và Recall

### 5.2. Đánh giá trên Testing Data

**Testing Data có đặc điểm:**

- Multi-label (một file có thể có nhiều nhạc cụ)
- Độ dài file khác nhau
- Không có trong training set

**Quy trình:**

1. Load audio file
2. Extract segments (sliding window)
3. Predict từng segment
4. **Weighted Average**: Segment có confidence cao hơn được ưu tiên
5. Final prediction

```python
# Weighted average aggregation
segment_weights = np.max(segment_probs, axis=1)  # Confidence
segment_weights = segment_weights / segment_weights.sum()
avg_probs = np.average(segment_probs, axis=0, weights=segment_weights)
```

### 5.3. Kết quả mong đợi

**Accuracy:** ~79-81% trên Testing Data

**Các class tốt:**

- Flute, Acoustic Guitar, Electric Guitar, Piano: Precision > 70%

**Các class khó:**

- Saxophone, Trumpet: Precision thấp (dễ nhầm với nhau)
- Organ, Clarinet: Precision thấp (ít data)

### 5.4. Confusion Matrix

Phân tích confusion matrix để xem:

- Class nào dễ nhầm với nhau
- Class nào cần cải thiện

**Ví dụ nhầm lẫn phổ biến:**

- Saxophone → Trumpet (và ngược lại)
- Violin → Voice
- Organ → Piano

============

## 💻 STEP 6: ỨNG DỤNG THỰC TẾ

### 6.1. Load Model

```python
import keras
import joblib

# Load model với custom_objects cho Focal Loss
model = keras.models.load_model(
    'IRMAS_Models/best_segment_cnn.keras',
    custom_objects={'focal_loss_fixed': focal_loss(gamma=2.0)}
)

# Load label encoder
label_encoder = joblib.load('IRMAS_Models/label_encoder_seg.joblib')
```

### 6.2. Quy trình Real-time Recognition

**Khi chạy ứng dụng thực tế, model sẽ làm gì để nhận biết nhạc cụ?**

#### 6.2.1. Quy trình đầy đủ

```
1. Input Audio (từ file hoặc microphone)
   ↓
2. Preprocessing (Resample, Normalize)
   ↓
3. Extract Segments (Sliding Window)
   ↓
4. Convert to Mel-Spectrogram
   ↓
5. Predict từng Segment
   ↓
6. Weighted Average Aggregation
   ↓
7. Final Prediction + Confidence
```

#### 6.2.2. Chi tiết từng bước

**Bước 1: Load Audio**

```python
# Từ file hoặc record từ microphone
audio, sr = librosa.load('audio.wav', sr=22050)
# Hoặc
audio = record_from_microphone()  # Record 3-5 giây
```

**Bước 2: Preprocessing**

```python
# Resample về 22050 Hz (nếu cần)
if sample_rate != 22050:
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050)

# Normalize về [-1, 1]
if audio.max() > 1.0 or audio.min() < -1.0:
    audio = audio / np.max(np.abs(audio))
```

**⚠️ Tại sao cần normalize audio?**

**Vấn đề:**

- Audio từ các nguồn khác nhau có amplitude khác nhau
- File A: amplitude [-0.5, 0.5]
- File B: amplitude [-2.0, 2.0]
- File C: amplitude [-0.1, 0.1]

**Nếu không normalize:**

- Model sẽ bị ảnh hưởng bởi volume (amplitude)
- File có volume cao → Mel-spectrogram có giá trị lớn → Model nghĩ là "quan trọng hơn"
- File có volume thấp → Mel-spectrogram có giá trị nhỏ → Model nghĩ là "ít quan trọng"

**Sau khi normalize:**

- Tất cả audio đều có amplitude [-1, 1]
- Model chỉ tập trung vào **pattern** (hình dạng sóng), không phụ thuộc vào **volume**
- Robust với các mức volume khác nhau

**Ví dụ:**

```
Trước normalize:
File A: [0.1, 0.2, 0.3, ...] → Mel-spec: giá trị nhỏ
File B: [0.5, 1.0, 1.5, ...] → Mel-spec: giá trị lớn
→ Model nghĩ File B "quan trọng hơn" (SAI!)

Sau normalize:
File A: [0.2, 0.4, 0.6, ...] → Mel-spec: giá trị tương đương
File B: [0.2, 0.4, 0.6, ...] → Mel-spec: giá trị tương đương
→ Model chỉ nhìn vào pattern (ĐÚNG!)
```

**Bước 3: Extract Segments (Sliding Window)**

```python
def extract_sliding_segments(audio, segment_duration=2.0, overlap=0.5):
    """
    Cắt audio thành segments với sliding window
    Overlap 50% → Cover toàn bộ audio
    """
    segments = []
    segment_samples = int(22050 * segment_duration)  # 44,100 samples
    hop = int(segment_samples * (1 - overlap))  # 22,050 samples (50% overlap)

    start = 0
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples]
        segments.append(segment)
        start += hop  # Nhảy 50% segment

    return segments
```

**Ví dụ với audio 5s:**

```
Audio: [==========] (5 giây)
       |--2s--|      ← Segment 1 (0.0s → 2.0s)
            |--2s--| ← Segment 2 (1.0s → 3.0s, overlap 50%)
                 |--2s--| ← Segment 3 (2.0s → 4.0s)
                      |--2s--| ← Segment 4 (3.0s → 5.0s)

Kết quả: 4 segments
```

**Bước 4: Convert to Mel-Spectrogram**

```python
def segment_to_mel(segment):
    """
    Chuyển mỗi segment → Mel-Spectrogram (128 × 87)
    """
    # 1. Tính Mel-Spectrogram (power spectrum)
    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=22050,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    # Output: Giá trị power (có thể rất lớn, ví dụ: 0 → 1000000)

    # 2. Chuyển sang decibel (dB) và normalize
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Output: Giá trị dB (thường từ -80 dB đến 0 dB)
    # ref=np.max → Normalize về [min, 0], với max = 0 dB

    return mel_spec_db

# Convert tất cả segments
mel_segments = [segment_to_mel(seg) for seg in segments]
mel_segments = np.array(mel_segments)[..., np.newaxis]  # Shape: (N, 128, 87, 1)
```

**⚠️ Tại sao cần `power_to_db` với `ref=np.max`?**

**Vấn đề với Mel-Spectrogram gốc:**

- Mel-spectrogram có giá trị **power** (có thể rất lớn: 0 → 1,000,000)
- Giá trị phân bố không đều → Khó cho model học
- Model dễ bị ảnh hưởng bởi giá trị tuyệt đối

**Giải pháp: `power_to_db` với `ref=np.max`:**

```python
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
```

**Cách hoạt động:**

1. Chuyển power → decibel (dB): `dB = 10 * log10(power)`
2. `ref=np.max`: Lấy max của mel-spectrogram làm reference
3. Normalize: `dB_normalized = dB - max_dB`
4. Kết quả: Giá trị từ `-∞` đến `0` dB (max = 0 dB)

**Ví dụ:**

```
Mel-Spectrogram gốc (power):
┌─────────────────┐
│ 1000  500  200 │ ← Giá trị lớn, không đều
│  500  300  100 │
│  200  100   50 │
└─────────────────┘

Sau power_to_db (ref=np.max):
┌─────────────────┐
│   0   -3   -7  │ ← Giá trị dB, normalize về [min, 0]
│  -3   -5   -10 │
│  -7  -10  -13  │
└─────────────────┘
```

**Lợi ích:**

- Giá trị trong khoảng hợp lý (thường -80 dB đến 0 dB)
- Normalize về cùng scale → Model học tốt hơn
- Tập trung vào **tương đối** (relative), không phụ thuộc vào **tuyệt đối** (absolute)
- Giống với cách tai người cảm nhận (logarithmic scale)

**Bước 5: Predict từng Segment**

```python
# Predict tất cả segments cùng lúc (batch processing)
segment_probs = model.predict(mel_segments, batch_size=32, verbose=0)
# Output: (N, 11) - N segments, 11 classes
# Mỗi hàng là probability distribution cho 1 segment
```

**Ví dụ với 4 segments:**

```
Segment 1: [0.01, 0.02, 0.05, 0.80, 0.03, ...] → Guitar (80%)
Segment 2: [0.01, 0.03, 0.04, 0.75, 0.05, ...] → Guitar (75%)
Segment 3: [0.02, 0.01, 0.06, 0.82, 0.02, ...] → Guitar (82%)
Segment 4: [0.01, 0.02, 0.05, 0.78, 0.04, ...] → Guitar (78%)
```

**Bước 6: Weighted Average Aggregation**

**Tại sao cần Weighted Average?**

- Không phải tất cả segments đều có chất lượng như nhau
- Segment có confidence cao → Đáng tin cậy hơn
- Segment có confidence thấp → Có thể bị nhiễu

**Công thức gốc (Weighted Average):**

```
P_final = Σᵢ (w_i · P_i) / Σᵢ w_i
```

Vì weights đã được normalize (Σᵢ w_i = 1), công thức đơn giản hóa thành:

```
P_final = Σᵢ (w_i · P_i)
```

Trong đó:

- `P_i`: Probability vector của segment i (shape: 11 classes)
- `w_i`: Weight của segment i (đã normalize, Σᵢ w_i = 1)
- `P_final`: Final probability vector

**Weight calculation:**

```
w_i = max(P_i) / Σⱼ max(P_j)
```

- `max(P_i)`: Confidence của segment i (probability cao nhất)
- Normalize để tổng weights = 1.0: `w_i = w_i / Σⱼ w_j`

**Cách tính:**

```python
# 1. Tính confidence của mỗi segment (max probability)
segment_weights = np.max(segment_probs, axis=1)
# Ví dụ: [0.80, 0.75, 0.82, 0.78] → Confidence của mỗi segment

# 2. Normalize weights (tổng = 1)
segment_weights = segment_weights / (segment_weights.sum() + 1e-10)
# Ví dụ: [0.25, 0.24, 0.26, 0.25] → Trọng số của mỗi segment

# 3. Weighted average: P_final = Σ(w_i · P_i)
avg_probs = np.average(segment_probs, axis=0, weights=segment_weights)
# Kết quả: [0.01, 0.02, 0.05, 0.79, 0.03, ...] → Final probabilities
```

**⚠️ Tại sao cần normalize weights?**

**Vấn đề:**

- Confidence của các segments: [0.80, 0.75, 0.82, 0.78]
- Tổng = 3.15 (không phải 1.0)
- Nếu dùng trực tiếp → Weighted average sẽ bị scale lên 3.15 lần (SAI!)

**Sau khi normalize:**

- [0.80, 0.75, 0.82, 0.78] → [0.25, 0.24, 0.26, 0.25]
- Tổng = 1.0 → Đúng với định nghĩa "trọng số" (weights)

**Ví dụ:**

```
Trước normalize:
Weights: [0.80, 0.75, 0.82, 0.78] (tổng = 3.15)
→ Weighted average sẽ bị scale lên 3.15 lần (SAI!)

Sau normalize:
Weights: [0.25, 0.24, 0.26, 0.25] (tổng = 1.0)
→ Weighted average đúng (ĐÚNG!)
```

**Lợi ích:**

- Đảm bảo tổng weights = 1.0 → Kết quả đúng
- Segment có confidence cao → Weight cao hơn
- Segment có confidence thấp → Weight thấp hơn

**Ví dụ cụ thể:**

```
Segment 1: Guitar 80% confidence → Weight 0.25
Segment 2: Guitar 75% confidence → Weight 0.24
Segment 3: Guitar 82% confidence → Weight 0.26 (cao nhất)
Segment 4: Guitar 78% confidence → Weight 0.25

Weighted Average:
Guitar = 0.25×0.80 + 0.24×0.75 + 0.26×0.82 + 0.25×0.78 = 0.79 (79%)
```

**Bước 7: Final Prediction**

```python
# Lấy class có probability cao nhất
predicted_idx = np.argmax(avg_probs)
predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
confidence = avg_probs[predicted_idx] * 100

# Top-3 predictions
top3_indices = np.argsort(avg_probs)[::-1][:3]
```

**Ví dụ kết quả:**

```
Predicted: Acoustic Guitar
Confidence: 79.23%

Top 3:
1. Acoustic Guitar: 79.23%
2. Electric Guitar: 12.45%
3. Piano: 5.32%
```

#### 6.2.3. Code đầy đủ

```python
def predict_audio_file(audio_path, model, label_encoder):
    """
    Predict nhạc cụ từ audio file
    """
    # 1. Load audio
    audio, sr = librosa.load(audio_path, sr=22050)

    # 2. Preprocessing
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
    if audio.max() > 1.0:
        audio = audio / np.max(np.abs(audio))

    # 3. Extract segments (sliding window, overlap 50%)
    segments = extract_sliding_segments(audio, segment_duration=2.0, overlap=0.5)

    # 4. Convert to mel-spectrogram
    mel_segments = [segment_to_mel(seg) for seg in segments]
    mel_segments = np.array(mel_segments)[..., np.newaxis]

    # 5. Predict
    segment_probs = model.predict(mel_segments, batch_size=32, verbose=0)

    # 6. Weighted average
    segment_weights = np.max(segment_probs, axis=1)  # Confidence
    segment_weights = segment_weights / (segment_weights.sum() + 1e-10)
    avg_probs = np.average(segment_probs, axis=0, weights=segment_weights)

    # 7. Final prediction
    predicted_idx = np.argmax(avg_probs)
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = avg_probs[predicted_idx] * 100

    return predicted_label, confidence, avg_probs
```

#### 6.2.4. Tại sao dùng Sliding Window + Weighted Average?

**Sliding Window:**

- Cover toàn bộ audio (không bỏ sót phần nào)
- Overlap 50% → Đảm bảo không mất thông tin ở ranh giới

**Weighted Average:**

- Segment có confidence cao → Đáng tin cậy hơn
- Segment có confidence thấp → Có thể bị nhiễu → Ít ảnh hưởng
- Kết quả robust hơn so với simple average

**Ví dụ so sánh:**

```
Simple Average:
Segment 1: Guitar 80% + Segment 2: Guitar 75% → Guitar 77.5%

Weighted Average:
Segment 1: Guitar 80% (weight 0.6) + Segment 2: Guitar 75% (weight 0.4)
→ Guitar 78% (Segment 1 có confidence cao hơn → weight cao hơn)
```

### 6.3. Test Model Script

File `test_model.py` cung cấp script test nhanh để kiểm tra:

- Model có load được không (với `custom_objects` cho Focal Loss)
- Label encoder có load được không
- Segment config có load được không
- Prediction với dummy data có hoạt động không

**Chạy test:**

```bash
python test_model.py
```

**Lưu ý:** Script này cần định nghĩa hàm `focal_loss` để load model thành công.

### 6.4. GUI Application

File `instrument_recognition_program.py` cung cấp:

- Giao diện đồ họa (Tkinter)
- Record audio từ microphone
- Load file audio
- Predict và hiển thị kết quả
- Top-3 predictions với confidence

**Chạy ứng dụng:**

```bash
python instrument_recognition_program.py
```

============

## 📝 TÓM TẮT CÁC BƯỚC

1. **Chuẩn bị dữ liệu**

   - Tải IRMAS dataset
   - Giải nén vào Google Drive
   - Mount Drive trong Colab

2. **Xử lý dữ liệu**

   - Load files từ các thư mục
   - Chia train/val/test ở cấp độ file
   - Trích xuất segments (random cho train, sliding cho val/test)
   - Convert sang mel-spectrogram

3. **Xây dựng model**

   - CNN với 4 Conv Blocks (32 → 64 → 128 → 256 filters)
   - Batch Normalization và Dropout để tránh overfitting
   - Focal Loss (gamma=2.0)

4. **Training**

   - Data augmentation (SpecAugment + Mixup)
   - Class weights để xử lý class imbalance
   - Callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
   - Train 100 epochs

5. **Đánh giá**

   - Test trên TestingData
   - Phân tích confusion matrix
   - Classification report

6. **Ứng dụng**
   - Load model và predict
   - Sử dụng trong GUI application

---

## ⚠️ LƯU Ý QUAN TRỌNG

1. **Data Leakage**: Luôn chia train/test ở cấp độ FILE trước khi cắt segments
2. **GPU Runtime**: Bắt buộc dùng GPU trong Colab để training nhanh
3. **Focal Loss**: Phải cung cấp `custom_objects` khi load model
4. **Segment Aggregation**: Dùng weighted average (không phải simple mean)
5. **Model Saving**: Lưu model vào Google Drive để không mất khi runtime disconnect

---

## 📁 CẤU TRÚC THƯ MỤC PROJECT

```
Musical_Instrument_Detection/
├── IRMAS_Training_CNN.ipynb       # Notebook training chính
├── instrument_recognition_program.py  # Ứng dụng GUI
├── test_model.py                  # Script test model
├── requirements.txt               # Dependencies
├── huongdan.md                   # File hướng dẫn này
├── README_DEMO.md                # Hướng dẫn demo
├── QUICK_START.md                # Quick start guide
├── IRMAS_Models/                 # Thư mục chứa models
│   ├── best_segment_cnn.keras
│   ├── label_encoder_seg.joblib
│   └── segment_config.joblib
└── IRMAS/                        # Dataset (sau khi tải và giải nén)
    ├── TrainingData/
    └── TestingData/
```

---

## 🔗 TÀI LIỆU THAM KHẢO

- **IRMAS Dataset**: [Link dataset - cần cập nhật link cụ thể]
- **Focal Loss Paper**: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
- **Librosa Documentation**: https://librosa.org/
- **TensorFlow/Keras Documentation**: https://www.tensorflow.org/api_docs

---

**Chúc bạn thành công với đồ án! 🎵🎸🎹**
