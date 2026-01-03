# 📝 SCRIPT CHI TIẾT - NGƯỜI 3: ĐÁNH GIÁ & ỨNG DỤNG

**Thời lượng:** 8-10 phút  
**Nội dung:** STEP 5 (Đánh giá Model) + STEP 6 (Ứng dụng thực tế)

---

## 🎯 MỤC TIÊU TRÌNH BÀY

1. Giải thích các metrics đánh giá model (Accuracy, Precision, Recall, F1-Score)
2. Trình bày kết quả trên Testing Data
3. Phân tích Confusion Matrix
4. Giải thích quy trình Real-time Recognition
5. Demo GUI Application

---

## 📊 PHẦN 1: STEP 5 - ĐÁNH GIÁ MODEL (4-5 phút)

### 1.0. Đánh giá trên Test Set (từ TrainingData) - 0.5 phút

**📝 ĐOẠN NÓI:**

> "Sau khi training xong, chúng ta cần đánh giá model. Có hai phần đánh giá: Đầu tiên là đánh giá trên Test Set được chia từ TrainingData. Đây là phần dữ liệu đã được tách ra từ đầu, không dùng trong quá trình training, nên có thể đánh giá khách quan. Chúng ta sử dụng hàm evaluate_with_aggregation, tức là cắt mỗi file thành nhiều segments bằng sliding window, predict từng segment, rồi sử dụng Weighted Average để kết hợp các dự đoán. Kết quả trên Test Set này thường là khoảng 84 đến 85 phần trăm accuracy, cho thấy model đã học tốt trên dữ liệu training."

> "Tuy nhiên, để đánh giá thực sự khách quan, chúng ta cần test trên TestingData riêng, đây là dataset hoàn toàn mới mà model chưa từng thấy. Đây mới là đánh giá quan trọng nhất."

---

### 1.1. Giới thiệu Metrics (2 phút)

**📝 ĐOẠN NÓI - MỞ ĐẦU:**

> "Sau khi training xong, chúng ta cần đánh giá xem model có hoạt động tốt không. Để làm điều này, chúng ta sử dụng 4 metrics chính: Accuracy, Precision, Recall, và F1-Score. Mỗi metric sẽ cho chúng ta biết một khía cạnh khác nhau về hiệu suất của model."

---

#### ✅ ĐOẠN NÓI 1: Accuracy (Độ chính xác)

**📝 ĐOẠN NÓI:**

> "Đầu tiên là Accuracy, hay còn gọi là độ chính xác. Accuracy cho chúng ta biết tổng quan: trong tất cả các mẫu test, model dự đoán đúng được bao nhiêu phần trăm. Công thức của Accuracy rất đơn giản: Accuracy bằng số mẫu dự đoán đúng chia cho tổng số mẫu. Ví dụ, nếu chúng ta có 100 mẫu test và model dự đoán đúng 80 mẫu, thì Accuracy sẽ là 80 phần trăm. Đây là metric dễ hiểu nhất, nhưng đôi khi nó không phản ánh đầy đủ hiệu suất của model, đặc biệt là khi có class imbalance, tức là một số nhạc cụ có nhiều mẫu hơn các nhạc cụ khác."

**Công thức:**

```
Accuracy = Số mẫu dự đoán đúng / Tổng số mẫu
```

---

#### ✅ ĐOẠN NÓI 2: Precision (Độ chính xác dự đoán)

**📝 ĐOẠN NÓI:**

> "Tiếp theo là Precision, hay độ chính xác dự đoán. Precision trả lời câu hỏi: trong số tất cả các mẫu mà model dự đoán là một nhạc cụ cụ thể, ví dụ như Guitar, thì có bao nhiêu phần trăm là đúng thực sự là Guitar? Precision được tính bằng số dự đoán đúng chia cho tổng số dự đoán của class đó. Ví dụ, nếu model dự đoán 50 mẫu là Guitar, nhưng chỉ có 40 mẫu thực sự là Guitar, thì Precision của Guitar sẽ là 40 chia 50, tức là 80 phần trăm. Precision cao có nghĩa là khi model nói một mẫu là Guitar, thì khả năng cao là nó đúng là Guitar, tức là model ít dự đoán sai."

**Công thức:**

```
Precision = Số dự đoán đúng class i / Tổng số dự đoán là class i
```

---

#### ✅ ĐOẠN NÓI 3: Recall (Độ nhạy)

**📝 ĐOẠN NÓI:**

> "Thứ ba là Recall, hay còn gọi là độ nhạy. Recall trả lời câu hỏi ngược lại với Precision: trong số tất cả các mẫu thực tế là một nhạc cụ cụ thể, ví dụ như Guitar, thì model tìm được bao nhiêu phần trăm? Recall được tính bằng số dự đoán đúng chia cho tổng số mẫu thực tế của class đó. Ví dụ, nếu có 60 mẫu thực tế là Guitar, nhưng model chỉ tìm được 45 mẫu, thì Recall của Guitar sẽ là 45 chia 60, tức là 75 phần trăm. Recall cao có nghĩa là model ít bỏ sót, tức là khi có một mẫu là Guitar, model sẽ tìm thấy nó. Đây là điều quan trọng trong thực tế, vì chúng ta không muốn bỏ sót nhạc cụ nào."

**Công thức:**

```
Recall = Số dự đoán đúng class i / Tổng số mẫu thực tế là class i
```

---

#### ✅ ĐOẠN NÓI 4: F1-Score (Trung bình điều hòa)

**📝 ĐOẠN NÓI:**

> "Cuối cùng là F1-Score. F1-Score là một metric kết hợp cả Precision và Recall. Nó được tính bằng công thức: 2 nhân với Precision nhân Recall, chia cho tổng của Precision và Recall. F1-Score là trung bình điều hòa của Precision và Recall, có nghĩa là nó cân bằng giữa hai metric này. Ví dụ, nếu Precision là 80 phần trăm và Recall là 75 phần trăm, thì F1-Score sẽ là khoảng 77 phần trăm. F1-Score cao có nghĩa là cả Precision và Recall đều tốt, tức là model vừa ít dự đoán sai, vừa ít bỏ sót. Đây là metric rất hữu ích khi chúng ta muốn đánh giá tổng thể hiệu suất của model."

**Công thức:**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

#### 📊 ĐOẠN NÓI 5: Macro Average và Weighted Average

**📝 ĐOẠN NÓI:**

> "Khi đánh giá model với nhiều classes, chúng ta có hai cách tính trung bình: Macro Average và Weighted Average. Macro Average là trung bình đơn giản của tất cả các classes, tức là chúng ta tính Precision, Recall, và F1 cho từng class, rồi lấy trung bình. Cách này đối xử công bằng với tất cả các classes, không quan tâm đến số lượng mẫu của mỗi class. Còn Weighted Average là trung bình có trọng số, tức là các classes có nhiều mẫu hơn sẽ có trọng số cao hơn. Cách này phản ánh tốt hơn hiệu suất thực tế của model, vì nó ưu tiên các classes có nhiều dữ liệu hơn. Trong đồ án này, chúng ta thường dùng Weighted Average để đánh giá, vì dataset có class imbalance, một số nhạc cụ có nhiều mẫu hơn các nhạc cụ khác."

---

### 1.2. Đánh giá trên Testing Data (1.5 phút)

**📝 ĐOẠN NÓI:**

> "Để đánh giá model một cách khách quan, chúng ta sử dụng Testing Data. Testing Data có một số đặc điểm quan trọng: thứ nhất, nó là multi-label, tức là một file có thể chứa nhiều nhạc cụ cùng lúc. Thứ hai, độ dài các file khác nhau, không đồng nhất. Và quan trọng nhất, Testing Data hoàn toàn tách biệt với Training Data, không có file nào xuất hiện trong cả hai tập, nên đánh giá này rất khách quan và phản ánh đúng khả năng của model trên dữ liệu mới."

> "Quy trình đánh giá như sau: Đầu tiên, chúng ta load một file audio từ Testing Data. Sau đó, chúng ta cắt file này thành các segments bằng sliding window với overlap 50 phần trăm, để đảm bảo cover toàn bộ audio. Tiếp theo, chúng ta predict từng segment và nhận được các vector probability. Sau đó, chúng ta sử dụng Weighted Average để kết hợp các dự đoán này thành một kết quả cuối cùng. Cuối cùng, chúng ta so sánh kết quả với nhãn thực tế để tính các metrics như Accuracy, Precision, Recall, và F1-Score."

> "Lưu ý: Vì Testing Data là multi-label, một file có thể có nhiều nhạc cụ. Trong đánh giá, chúng ta thường lấy label đầu tiên làm ground truth chính, hoặc có thể đánh giá theo cách khác tùy vào yêu cầu."

---

### 1.3. Kết quả thực nghiệm (1 phút)

**📝 ĐOẠN NÓI:**

> "Sau khi đánh giá trên toàn bộ Testing Data, chúng ta thu được kết quả như sau: Accuracy của model là khoảng 79 đến 81 phần trăm. Đây là một kết quả khá tốt, đặc biệt là khi xét đến việc dataset có class imbalance và một số nhạc cụ có âm thanh tương tự nhau."

> "Khi phân tích chi tiết từng class, chúng ta thấy rằng các nhạc cụ phổ biến như Flute, Acoustic Guitar, Electric Guitar, và Piano có Precision cao hơn 70 phần trăm. Điều này có nghĩa là model nhận diện tốt các nhạc cụ này. Tuy nhiên, một số nhạc cụ khác lại gặp khó khăn. Ví dụ, Saxophone và Trumpet có Precision thấp, vì chúng đều là kèn và có âm thanh tương tự nhau, nên model dễ nhầm lẫn. Ngoài ra, Organ và Clarinet cũng có Precision thấp, nhưng lý do ở đây là do ít data trong dataset, nên model chưa học tốt các nhạc cụ này."

---

### 1.4. Classification Report và Confusion Matrix (0.5 phút)

**📝 ĐOẠN NÓI:**

> "Để đánh giá chi tiết, chúng ta sử dụng Classification Report, cho biết Precision, Recall, và F1-Score của từng class. Điều này giúp chúng ta biết class nào model nhận diện tốt, class nào cần cải thiện."

> "Ngoài ra, chúng ta cũng sử dụng Confusion Matrix để hiểu rõ hơn về những lỗi mà model mắc phải. Confusion Matrix cho chúng ta biết class nào dễ nhầm với class nào. Từ đó, chúng ta có thể xác định được những điểm cần cải thiện. Ví dụ, chúng ta thấy rằng Saxophone thường bị nhầm với Trumpet và ngược lại, vì cả hai đều là kèn. Violin đôi khi bị nhầm với Voice, có thể do âm thanh tương tự. Organ cũng dễ bị nhầm với Piano, vì cả hai đều là nhạc cụ phím."

> "Từ những phân tích này, chúng ta có thể kết luận rằng model hoạt động tốt với các nhạc cụ phổ biến và có đặc trưng rõ ràng, nhưng cần cải thiện cho các nhạc cụ có âm thanh tương tự hoặc có ít data. Để cải thiện, chúng ta có thể thu thập thêm data cho các class yếu, hoặc sử dụng các kỹ thuật như data augmentation hoặc fine-tuning."

---

## 💻 PHẦN 2: STEP 6 - ỨNG DỤNG THỰC TẾ (4-5 phút)

### 2.1. Load Model (0.5 phút)

**Code:**

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

**Lưu ý quan trọng:**

- Phải cung cấp `custom_objects` cho Focal Loss
- Nếu không có → Model sẽ không load được

---

### 2.2. Quy trình Real-time Recognition (2.5 phút)

**📝 ĐOẠN NÓI - TỔNG QUAN:**

> "Bây giờ tôi sẽ giải thích quy trình khi model nhận diện nhạc cụ trong thực tế. Quy trình này gồm 7 bước chính: Đầu tiên là input audio, có thể từ file hoặc từ microphone. Tiếp theo là preprocessing, tức là xử lý sơ bộ audio. Sau đó, chúng ta cắt audio thành các segments bằng sliding window. Mỗi segment được chuyển đổi sang Mel-Spectrogram. Tiếp theo, model sẽ predict từng segment. Sau đó, chúng ta sử dụng Weighted Average để kết hợp các dự đoán. Cuối cùng, chúng ta có kết quả cuối cùng cùng với confidence."

**Sơ đồ quy trình:**

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

---

#### Bước 1: Load Audio (0.3 phút)

**📝 ĐOẠN NÓI:**

> "Bước đầu tiên rất đơn giản, chúng ta load audio từ file hoặc record từ microphone. Nếu từ file, chúng ta dùng thư viện librosa để load. Nếu từ microphone, chúng ta record khoảng 3 đến 5 giây. Audio này sẽ là input cho các bước tiếp theo."

---

#### Bước 2: Preprocessing (0.5 phút)

**📝 ĐOẠN NÓI:**

> "Bước thứ hai là preprocessing, tức là xử lý sơ bộ audio. Tại sao cần bước này? Vấn đề là audio từ các nguồn khác nhau có amplitude khác nhau. Ví dụ, một file có thể có amplitude từ -0.5 đến 0.5, còn file khác có thể có amplitude từ -2.0 đến 2.0. Nếu không normalize, model sẽ bị ảnh hưởng bởi volume, tức là file có volume cao sẽ được model nghĩ là quan trọng hơn, điều này hoàn toàn sai. Sau khi normalize, tất cả audio đều có amplitude từ -1 đến 1, và model chỉ tập trung vào pattern, tức là hình dạng sóng, không phụ thuộc vào volume. Điều này làm cho model robust hơn với các mức volume khác nhau."

> "Ngoài ra, chúng ta cũng cần resample audio về 22050 Hz nếu sample rate khác, để đảm bảo tính nhất quán với dữ liệu training."

---

#### Bước 3: Extract Segments - Sliding Window (0.5 phút)

**📝 ĐOẠN NÓI:**

> "Bước thứ ba là cắt audio thành các segments bằng sliding window. Tại sao dùng sliding window? Vì chúng ta muốn cover toàn bộ audio, không bỏ sót phần nào. Với overlap 50 phần trăm, chúng ta đảm bảo không mất thông tin ở ranh giới giữa các segments. Ví dụ, với một audio dài 5 giây, chúng ta sẽ có 4 segments, mỗi segment dài 2 giây, và các segments này overlap 50 phần trăm với nhau. Segment đầu tiên từ 0 đến 2 giây, segment thứ hai từ 1 đến 3 giây, segment thứ ba từ 2 đến 4 giây, và segment cuối cùng từ 3 đến 5 giây. Như vậy, mọi phần của audio đều được cover."

---

#### Bước 4: Convert to Mel-Spectrogram (0.4 phút)

**📝 ĐOẠN NÓI:**

> "Bước thứ tư là chuyển đổi mỗi segment sang Mel-Spectrogram. Đây là bước quan trọng vì model của chúng ta được train trên Mel-Spectrogram, không phải trên raw audio. Chúng ta sử dụng thư viện librosa để tính Mel-Spectrogram với các tham số: 128 mel bands, FFT window size 2048, và hop length 512. Sau đó, chúng ta chuyển từ power spectrum sang decibel bằng hàm power_to_db với ref bằng max. Điều này rất quan trọng vì Mel-Spectrogram gốc có giá trị power có thể rất lớn, từ 0 đến hàng triệu. Sau khi chuyển sang dB và normalize, giá trị sẽ nằm trong khoảng hợp lý, từ -80 dB đến 0 dB, và model sẽ học tốt hơn với các giá trị này."

---

#### Bước 5: Predict từng Segment (0.3 phút)

**📝 ĐOẠN NÓI:**

> "Bước thứ năm là predict từng segment. Chúng ta đưa tất cả các Mel-Spectrograms vào model cùng lúc, sử dụng batch processing để tăng tốc độ. Model sẽ trả về một vector probability cho mỗi segment, với 11 giá trị tương ứng với 11 loại nhạc cụ. Ví dụ, một segment có thể có vector [0.01, 0.02, 0.05, 0.80, 0.03, ...], trong đó giá trị 0.80 ở vị trí thứ 4 tương ứng với Guitar, có nghĩa là model dự đoán segment này là Guitar với confidence 80 phần trăm."

---

#### Bước 6: Weighted Average Aggregation (0.8 phút)

**📝 ĐOẠN NÓI - Weighted Average (QUAN TRỌNG NHẤT):**

> "Sau khi predict từng segment, chúng ta có nhiều kết quả dự đoán khác nhau. Câu hỏi đặt ra là: làm sao để kết hợp các dự đoán này thành một kết quả cuối cùng? Chúng ta không thể đơn giản lấy trung bình cộng, vì không phải tất cả các segments đều có chất lượng như nhau. Một số segment có confidence cao, tức là model rất chắc chắn về dự đoán, còn một số segment có confidence thấp, có thể bị nhiễu hoặc không rõ ràng. Vì vậy, chúng ta sử dụng Weighted Average, tức là trung bình có trọng số. Segment nào có confidence cao hơn sẽ được gán trọng số cao hơn, và ngược lại."

> "Cách tính như sau: Đầu tiên, chúng ta lấy confidence của mỗi segment, tức là giá trị probability cao nhất trong vector dự đoán của segment đó. Ví dụ, nếu một segment có vector dự đoán là [0.01, 0.02, 0.05, 0.80, 0.03, ...], thì confidence của nó là 0.80, tức là 80 phần trăm. Sau đó, chúng ta normalize các confidence này để tổng của chúng bằng 1. Điều này rất quan trọng, vì nếu không normalize, kết quả sẽ bị scale lên và không chính xác. Ví dụ, nếu chúng ta có 4 segments với confidence lần lượt là 0.80, 0.75, 0.82, và 0.78, thì tổng là 3.15. Nếu dùng trực tiếp, kết quả sẽ bị nhân lên 3.15 lần, điều này sai. Sau khi normalize, các trọng số sẽ là 0.25, 0.24, 0.26, và 0.25, tổng bằng 1, đúng như mong muốn."

> "Cuối cùng, chúng ta tính weighted average bằng cách nhân mỗi vector probability với trọng số tương ứng, rồi cộng lại. Công thức là: P_final bằng tổng của w_i nhân P_i, trong đó w_i là trọng số đã normalize, và P_i là vector probability của segment thứ i. Ví dụ cụ thể: nếu segment 1 có confidence 80 phần trăm và trọng số 0.25, segment 2 có confidence 75 phần trăm và trọng số 0.24, segment 3 có confidence 82 phần trăm và trọng số 0.26, và segment 4 có confidence 78 phần trăm và trọng số 0.25, thì kết quả cuối cùng cho Guitar sẽ là 0.25 nhân 0.80, cộng 0.24 nhân 0.75, cộng 0.26 nhân 0.82, cộng 0.25 nhân 0.78, bằng khoảng 0.79, tức là 79 phần trăm. Đây là cách chúng ta kết hợp các dự đoán từ nhiều segments thành một kết quả cuối cùng, và cách này robust hơn nhiều so với việc chỉ lấy trung bình đơn giản."

**Công thức:**

```
P_final = Σᵢ (w_i · P_i)
w_i = max(P_i) / Σⱼ max(P_j)  (sau khi normalize)
```

**Code:**

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

---

#### Bước 7: Final Prediction (0.2 phút)

**📝 ĐOẠN NÓI:**

> "Bước cuối cùng là lấy kết quả cuối cùng. Chúng ta tìm class có probability cao nhất trong vector weighted average. Class này chính là nhạc cụ được dự đoán. Chúng ta cũng có thể lấy top 3 predictions để người dùng biết các khả năng khác. Ví dụ, kết quả có thể là: Acoustic Guitar với confidence 79.23 phần trăm, Electric Guitar với 12.45 phần trăm, và Piano với 5.32 phần trăm. Đây chính là kết quả cuối cùng mà model trả về."

---

### 2.3. Code đầy đủ (0.5 phút)

**Hàm predict_audio_file:**

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

---

### 2.4. GUI Application - Demo (1 phút)

**📝 ĐOẠN NÓI:**

> "Để ứng dụng model vào thực tế, chúng ta đã xây dựng một ứng dụng GUI bằng Tkinter. Ứng dụng này cho phép người dùng record audio từ microphone hoặc load file audio, sau đó nhận diện nhạc cụ. Khi người dùng click nút 'Nhận dạng', ứng dụng sẽ thực hiện toàn bộ quy trình 7 bước mà tôi vừa giải thích, và hiển thị kết quả top 3 predictions cùng với confidence. Điều này giúp người dùng không chỉ biết nhạc cụ được dự đoán, mà còn biết các khả năng khác và mức độ chắc chắn của model."

> "Kết quả thực tế cho thấy model hoạt động khá tốt. Chúng ta có thể nhận diện nhạc cụ từ microphone hoặc từ file audio với accuracy khoảng 79 đến 81 phần trăm trên Testing Data. Đây là một kết quả đáng khích lệ, đặc biệt là khi xét đến độ phức tạp của bài toán và sự đa dạng của dữ liệu."

---

## 📋 TÓM TẮT CÁC CÔNG THỨC CẦN NÓI

### ✅ BẮT BUỘC PHẢI NÓI:

1. **Accuracy:**

   ```
   Accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **Precision:**

   ```
   Precision = TP / (TP + FP)
   ```

3. **Recall:**

   ```
   Recall = TP / (TP + FN)
   ```

4. **F1-Score:**

   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```

5. **Weighted Average (QUAN TRỌNG NHẤT):**
   ```
   P_final = Σᵢ (w_i · P_i)
   w_i = max(P_i) / Σⱼ max(P_j)
   ```

### ⚠️ CÓ THỂ NÓI (NẾU CÓ THỜI GIAN):

- Macro Average
- Weighted Average cho metrics (Precision, Recall, F1)

---

## 🎤 TIPS KHI TRÌNH BÀY

### 1. Phần Metrics (STEP 5)

- **Nhấn mạnh:** Giải thích rõ sự khác biệt giữa Precision và Recall
- **Ví dụ:** Dùng ví dụ cụ thể (Guitar, Piano) để dễ hiểu
- **Slide:** Có thể vẽ bảng Confusion Matrix để minh họa

### 2. Phần Weighted Average (QUAN TRỌNG)

- **Nhấn mạnh:** Tại sao cần normalize weights
- **Ví dụ:** Dùng ví dụ 4 segments với confidence khác nhau
- **Slide:** Có thể vẽ sơ đồ minh họa quá trình aggregation

### 3. Phần Demo (STEP 6)

- **Chuẩn bị:** Test trước GUI application để đảm bảo hoạt động
- **Audio mẫu:** Chuẩn bị 2-3 file audio mẫu (Guitar, Piano, Flute)
- **Nếu lỗi:** Có thể giải thích quy trình thay vì demo trực tiếp

### 4. Timing

- **STEP 5:** 4-5 phút (Metrics: 2 phút, Kết quả: 1.5 phút, Confusion Matrix: 0.5 phút)
- **STEP 6:** 4-5 phút (Load Model: 0.5 phút, Quy trình: 2.5 phút, Code: 0.5 phút, Demo: 1 phút)
- **Dự phòng:** Để lại 1-2 phút cho Q&A

### 5. Câu hỏi thường gặp

- **Q: Tại sao dùng Weighted Average thay vì Simple Average?**
  - A: Segment có confidence cao đáng tin cậy hơn, nên được ưu tiên
- **Q: Tại sao Accuracy chỉ ~80%?**
  - A: Dataset có class imbalance, một số nhạc cụ dễ nhầm với nhau (Saxophone/Trumpet)
- **Q: Model có thể nhận diện nhiều nhạc cụ cùng lúc không?**
  - A: Hiện tại model chỉ nhận diện 1 nhạc cụ chính (top-1), nhưng có thể mở rộng để nhận diện multi-label

---

## ✅ CHECKLIST TRƯỚC KHI QUAY

- [ ] Đã đọc kỹ script này
- [ ] Đã hiểu rõ các công thức (Accuracy, Precision, Recall, F1, Weighted Average)
- [ ] Đã test GUI application trước
- [ ] Đã chuẩn bị audio mẫu để demo
- [ ] Đã chuẩn bị slide (nếu có)
- [ ] Đã luyện tập trình bày 2-3 lần
- [ ] Đã kiểm tra thời gian (8-10 phút)

---

## 📌 LƯU Ý CUỐI CÙNG

1. **Tự tin:** Bạn đã hiểu rõ phần này, cứ trình bày tự nhiên
2. **Giải thích rõ:** Đặc biệt là Weighted Average - đây là điểm khác biệt của đồ án
3. **Demo nếu có thể:** Demo GUI application sẽ rất ấn tượng
4. **Nếu quên:** Có thể nhìn vào slide hoặc notebook để nhắc lại

**Chúc bạn trình bày thành công! 🎉**
