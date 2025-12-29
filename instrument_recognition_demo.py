"""
Musical Instrument Recognition Demo
Sử dụng model đã train trên IRMAS dataset để nhận dạng nhạc cụ từ audio
"""

import tkinter as tk
from tkinter import messagebox as msb
import tkinter.filedialog as fd
import queue
import threading
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
import joblib
from tensorflow import keras
from scipy.ndimage import zoom
from PIL import Image, ImageTk
import os

# Tắt TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Chỉ hiển thị ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Tối ưu với oneDNN

class InstrumentRecognitionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Queue để chứa audio data
        self.q = queue.Queue()
        
        # Biến trạng thái
        self.recording = False
        self.file_exists = False
        self.audio_data = None
        self.sample_rate = None
        self.current_file = None
        
        # Load models và label encoders
        self.cnn_model = None
        self.svm_pipeline = None
        self.cnn_label_encoder = None
        self.svm_label_encoder = None
        self.segment_config = None
        self.instrument_names = {}
        self.load_models()
        
        # Mel-spectrogram parameters (phải khớp với lúc training)
        self.sr = 22050  # Sample rate
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.n_mfcc = 40  # Cho SVM features
        
        # Segment-based parameters (load từ config hoặc default)
        if self.segment_config:
            self.segment_duration = self.segment_config.get('segment_duration', 2.0)
            self.segment_overlap = self.segment_config.get('segment_overlap', 0.5)
        else:
            self.segment_duration = 2.0  # Default từ training
            self.segment_overlap = 0.5
        self.segment_samples = int(self.sr * self.segment_duration)
        
        # Phương pháp mặc định
        self.prediction_method = 'CNN'  # 'CNN' hoặc 'SVM'
        
        # Tên đầy đủ của các nhạc cụ
        self.setup_instrument_names()
        
        # Setup UI
        self.setup_ui()
        
    def setup_instrument_names(self):
        # Mapping nhạc cụ
        self.instrument_names = {
            'cel': 'Cello',
            'cla': 'Clarinet',
            'flu': 'Flute',
            'gac': 'Acoustic Guitar',
            'gel': 'Electric Guitar',
            'org': 'Organ',
            'pia': 'Piano',
            'sax': 'Saxophone',
            'tru': 'Trumpet',
            'vio': 'Violin',
            'voi': 'Voice'
        }
    
    def load_models(self):
        # Load cả CNN và SVM models từ thư mục IRMAS_Models
        try:
            # Load CNN model
            cnn_model_path = "IRMAS_Models/best_segment_cnn.keras"
            cnn_label_encoder_path = "IRMAS_Models/label_encoder_seg.joblib"
            
            if os.path.exists(cnn_model_path):
                self.cnn_model = keras.models.load_model(cnn_model_path)
                self.cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                print(f"CNN model loaded. Input shape: {self.cnn_model.input_shape}")
            else:
                print(f"Warning: CNN model not found: {cnn_model_path}")
            
            if os.path.exists(cnn_label_encoder_path):
                self.cnn_label_encoder = joblib.load(cnn_label_encoder_path)
                print(f"CNN label encoder loaded. Classes: {self.cnn_label_encoder.classes_}")
            else:
                print(f"Warning: CNN label encoder not found: {cnn_label_encoder_path}")
            
            # Load SVM pipeline
            svm_model_path = "IRMAS_Models/svm_instrument_model.joblib"
            svm_label_encoder_path = "IRMAS_Models/label_encoder_svm.joblib"
            
            if os.path.exists(svm_model_path):
                self.svm_pipeline = joblib.load(svm_model_path)
                print(f"SVM pipeline loaded. Keys: {self.svm_pipeline.keys() if isinstance(self.svm_pipeline, dict) else 'N/A'}")
            else:
                print(f"Warning: SVM model not found: {svm_model_path}")
            
            if os.path.exists(svm_label_encoder_path):
                self.svm_label_encoder = joblib.load(svm_label_encoder_path)
                print(f"SVM label encoder loaded. Classes: {self.svm_label_encoder.classes_}")
            else:
                # Nếu không có file riêng, lấy từ pipeline
                if isinstance(self.svm_pipeline, dict) and 'label_encoder' in self.svm_pipeline:
                    self.svm_label_encoder = self.svm_pipeline['label_encoder']
                    print("SVM label encoder loaded from pipeline")
                else:
                    print(f"Warning: SVM label encoder not found")
            
            # Load segment config
            config_path = "IRMAS_Models/segment_config.joblib"
            if os.path.exists(config_path):
                self.segment_config = joblib.load(config_path)
                print(f"Segment config loaded: {self.segment_config}")
            
            # Kiểm tra xem có model nào được load không
            if self.cnn_model is None and self.svm_pipeline is None:
                msb.showerror("Error", "No models found! Please check IRMAS_Models folder.")
                return
            
            msb.showinfo("Success", "Models loaded successfully!")
            
        except Exception as e:
            msb.showerror("Error", f"Failed to load models: {str(e)}")
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_ui(self):
        # Thiết lập giao diện
        self.title('Musical Instrument Recognition Demo')
        self.geometry('900x750')
        
        # Canvas để hiển thị waveform và spectrogram (tăng kích thước để có không gian cho labels)
        self.cvs_figure = tk.Canvas(self, width=700, height=500, relief=tk.SUNKEN, border=1, bg='white')
        
        # Tạo các LabelFrame
        lblf_controls = tk.LabelFrame(self, text="Controls", padx=5, pady=5)
        lblf_results = tk.LabelFrame(self, text="Recognition Results", padx=5, pady=5)
        lblf_status = tk.LabelFrame(self, text="Status", padx=5, pady=5)
        lblf_method = tk.LabelFrame(self, text="Prediction Method", padx=5, pady=5)
        
        # Method selection
        self.method_var = tk.StringVar(value='CNN')
        rb_cnn = tk.Radiobutton(lblf_method, text='CNN (Segment-based)', 
                               variable=self.method_var, value='CNN',
                               command=self.on_method_changed)
        rb_svm = tk.Radiobutton(lblf_method, text='SVM (Handcrafted)', 
                               variable=self.method_var, value='SVM',
                               command=self.on_method_changed)
        
        # Buttons trong controls frame
        btn_open = tk.Button(lblf_controls, text='Open File', width=12, command=self.open_file)
        btn_record = tk.Button(lblf_controls, text='Record', width=12, 
                              command=lambda: self.threading_rec(1))
        btn_stop = tk.Button(lblf_controls, text='Stop', width=12, 
                            command=lambda: self.threading_rec(2))
        btn_play = tk.Button(lblf_controls, text='Play', width=12, 
                            command=lambda: self.threading_rec(3))
        btn_predict = tk.Button(lblf_controls, text='Predict', width=12, 
                               command=self.predict_instrument, 
                               bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'))
        
        # Status label
        self.lbl_status = tk.Label(lblf_status, text="Ready", fg='blue', 
                                   font=('Arial', 10))
        
        # Results display
        self.lbl_result = tk.Label(lblf_results, text="No prediction yet", 
                                   font=('Arial', 12, 'bold'), fg='darkblue')
        self.lbl_confidence = tk.Label(lblf_results, text="", 
                                       font=('Arial', 10))
        
        # Top-3 results
        self.lbl_top3_title = tk.Label(lblf_results, text="Top 3 Predictions:", 
                                       font=('Arial', 10, 'bold'))
        self.lbl_top3 = tk.Label(lblf_results, text="", 
                                font=('Arial', 9), justify=tk.LEFT)
        
        # Layout method selection
        rb_cnn.pack(anchor='w', padx=5, pady=2)
        rb_svm.pack(anchor='w', padx=5, pady=2)
        
        # Layout controls
        btn_open.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        btn_record.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        btn_stop.grid(row=2, column=0, padx=5, pady=5, sticky='ew')
        btn_play.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
        btn_predict.grid(row=4, column=0, padx=5, pady=10, sticky='ew')
        
        # Layout status
        self.lbl_status.pack(pady=5)
        
        # Layout results
        self.lbl_result.pack(pady=5)
        self.lbl_confidence.pack(pady=2)
        self.lbl_top3_title.pack(pady=(10, 2))
        self.lbl_top3.pack(pady=2)
        
        # Main layout
        self.cvs_figure.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky='nsew')
        lblf_method.grid(row=0, column=1, padx=5, pady=5, sticky='n')
        lblf_controls.grid(row=1, column=1, padx=5, pady=5, sticky='s')  # Stick về dưới
        lblf_status.grid(row=2, column=0, padx=5, pady=5, sticky='ew', columnspan=2)
        lblf_results.grid(row=3, column=0, padx=5, pady=5, sticky='nsew', columnspan=2)
        
        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)  # Row 0 mở rộng để đẩy Controls xuống
        self.grid_rowconfigure(1, weight=0)  # Row 1 không mở rộng
        self.grid_rowconfigure(3, weight=1)  # Row 3 (results) mở rộng
        self.grid_columnconfigure(0, weight=1)
    
    def on_method_changed(self):
        # Callback khi thay đổi phương pháp prediction
        self.prediction_method = self.method_var.get()
        method_name = "CNN (Segment-based)" if self.prediction_method == 'CNN' else "SVM (Handcrafted)"
        self.update_status(f"Method changed to: {method_name}")
        
    def callback(self, indata, frames, time, status):
        # Callback function cho sounddevice recording
        self.q.put(indata.copy())
    
    def threading_rec(self, mode):
        # Xử lý recording/stop/play trong thread riêng
        if mode == 1:  # Record
            if self.recording:
                msb.showwarning("Warning", "Already recording!")
                return
            t1 = threading.Thread(target=self.record_audio)
            t1.daemon = True
            t1.start()
        elif mode == 2:  # Stop
            self.recording = False
            self.update_status("Recording stopped")
        elif mode == 3:  # Play
            self.play_audio()
    
    def record_audio(self):
        # Thu âm từ microphone
        self.recording = True
        self.update_status("Recording... Speak into the microphone")
        self.current_file = "recorded_audio.wav"
        
        try:
            with sf.SoundFile(self.current_file, mode='w', samplerate=self.sr, 
                            channels=1, subtype='PCM_16') as file:
                with sd.InputStream(samplerate=self.sr, channels=1, 
                                  callback=self.callback):
                    while self.recording:
                        self.file_exists = True
                        try:
                            # Timeout để tránh block vô hạn nếu có lỗi
                            file.write(self.q.get(timeout=1.0))
                        except queue.Empty:
                            # Nếu queue rỗng, tiếp tục loop
                            continue
            
            # Load audio sau khi ghi xong
            self.audio_data, self.sample_rate = sf.read(self.current_file, dtype='float32')
            self.update_status("Recording finished. Ready to predict.")
            self.draw_waveform()
            msb.showinfo("Recording", "Recording finished successfully!")
            
        except Exception as e:
            msb.showerror("Error", f"Recording failed: {str(e)}")
            self.update_status("Recording failed")
            self.recording = False
    
    def open_file(self):
        # Mở file audio từ disk
        filetypes = (("Wave files", "*.wav"), ("All files", "*.*"))
        filename = fd.askopenfilename(title="Open audio file", filetypes=filetypes)
        
        if filename:
            try:
                self.current_file = filename
                # Load audio với librosa để đảm bảo đúng format
                self.audio_data, self.sample_rate = librosa.load(filename, sr=None)
                self.file_exists = True
                self.update_status(f"File loaded: {os.path.basename(filename)}")
                self.draw_waveform()
                msb.showinfo("Success", "File loaded successfully!")
            except Exception as e:
                msb.showerror("Error", f"Failed to load file: {str(e)}")
                self.update_status("File loading failed")
    
    def play_audio(self):
        # Phát lại audio đã load
        if not self.file_exists or self.audio_data is None:
            msb.showerror("Error", "No audio to play. Please record or open a file first.")
            return
        
        try:
            self.update_status("Playing audio...")
            sd.play(self.audio_data, self.sample_rate)
            sd.wait()
            self.update_status("Playback finished")
        except Exception as e:
            msb.showerror("Error", f"Playback failed: {str(e)}")
            self.update_status("Playback failed")
    
    def draw_waveform(self):
        # Vẽ waveform và spectrogram trên canvas
        if self.audio_data is None:
            return
        
        self.cvs_figure.delete(tk.ALL)
        width = self.cvs_figure.winfo_width()
        height = self.cvs_figure.winfo_height()
        
        if width <= 1 or height <= 1:
            width = 700
            height = 500
        
        # Chia canvas thành 2 phần: waveform (trên) và spectrogram (dưới)
        waveform_height = height // 2
        spectrogram_height = height // 2
        
        # WAVEFORM
        data_len = len(self.audio_data)
        step = max(1, data_len // width)
        y_center = waveform_height // 2
        
        for x in range(width - 1):
            idx1 = min(int(x * step), data_len - 1)
            idx2 = min(int((x + 1) * step), data_len - 1)
            
            y1 = int(y_center - self.audio_data[idx1] * (waveform_height // 2 - 10))
            y2 = int(y_center - self.audio_data[idx2] * (waveform_height // 2 - 10))
            
            self.cvs_figure.create_line(x, y1, x + 1, y2, fill='green', width=1)
        
        # Vẽ đường phân cách
        self.cvs_figure.create_line(0, waveform_height, width, waveform_height, fill='gray', width=2)
        
        # SPECTROGRAM
        try:
            # Tính mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=self.audio_data,
                sr=self.sample_rate if self.sample_rate else self.sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize về [0, 255]
            mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-10)
            mel_spec_normalized = (mel_spec_normalized * 255).astype(np.uint8)
            
            spec_height, spec_width = mel_spec_normalized.shape
            target_width = width
            target_height = spectrogram_height
            
            # Tạo colormap (xanh -> vàng -> đỏ)
            def apply_colormap(intensity_array):
                intensity = intensity_array / 255.0
                # Jet colormap
                r = np.zeros_like(intensity)
                g = np.zeros_like(intensity)
                b = np.zeros_like(intensity)
                
                # Blue to cyan
                mask1 = intensity < 0.25
                r[mask1] = 0
                g[mask1] = intensity[mask1] * 4 * 255
                b[mask1] = 255
                
                # Cyan to green
                mask2 = (intensity >= 0.25) & (intensity < 0.5)
                r[mask2] = 0
                g[mask2] = 255
                b[mask2] = 255 - (intensity[mask2] - 0.25) * 4 * 255
                
                # Green to yellow
                mask3 = (intensity >= 0.5) & (intensity < 0.75)
                r[mask3] = (intensity[mask3] - 0.5) * 4 * 255
                g[mask3] = 255
                b[mask3] = 0
                
                # Yellow to red
                mask4 = intensity >= 0.75
                r[mask4] = 255
                g[mask4] = 255 - (intensity[mask4] - 0.75) * 4 * 255
                b[mask4] = 0
                
                return np.stack([r, g, b], axis=-1).astype(np.uint8)
            
            # Resize spectrogram về kích thước target
            zoom_factors = (target_height / spec_height, target_width / spec_width)
            mel_spec_resized = zoom(mel_spec_normalized, zoom_factors, order=1)
            
            # Đảo ngược để tần số thấp
            mel_spec_resized = np.flipud(mel_spec_resized)
            
            # Áp dụng colormap
            colored_spec = apply_colormap(mel_spec_resized)
            
            # Tạo PIL Image từ numpy array
            img = Image.fromarray(colored_spec, 'RGB')
            if img.size != (target_width, target_height):
                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Convert sang PhotoImage để hiển thị trên canvas
            photo = ImageTk.PhotoImage(image=img)
            
            # Hiển thị image trên canvas
            spectrogram_y = waveform_height
            self.cvs_figure.create_image(0, spectrogram_y, anchor='nw', image=photo)
            
            # Lưu reference để tránh garbage collection
            self.cvs_figure.spectrogram_image = photo
        except Exception as e:
            print(f"Error drawing spectrogram: {e}")
            import traceback
            traceback.print_exc()
    
    def extract_sliding_segments(self, audio):
        """
        Cắt audio thành các segments với sliding window và overlap.
        Dùng cho prediction (giống như trong training notebook).
        """
        segments = []
        audio_length = len(audio)
        
        # Pad nếu audio quá ngắn
        if audio_length < self.segment_samples:
            audio = np.pad(audio, (0, self.segment_samples - audio_length))
            audio_length = len(audio)
        
        # Tính hop size với overlap 50%
        hop = int(self.segment_samples * (1 - self.segment_overlap))
        start = 0
        
        # Extract segments với sliding window
        while start + self.segment_samples <= audio_length:
            segment = audio[start:start + self.segment_samples]
            segments.append(segment)
            start += hop
        
        # Thêm segment cuối nếu còn dư
        if start < audio_length:
            segment = audio[-self.segment_samples:]
            if len(segments) == 0 or not np.array_equal(segment, segments[-1]):
                segments.append(segment)
        
        return segments
    
    def segment_to_mel(self, segment):
        """
        Chuyển một segment thành mel spectrogram.
        Khớp với hàm segment_to_mel trong training notebook.
        """
        mel_spec = librosa.feature.melspectrogram(
            y=segment,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_handcrafted_features(self, audio):
        """
        Trích xuất handcrafted features cho SVM.
        Khớp với hàm extract_handcrafted_features trong training notebook.
        """
        features = []
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        for stat in [np.mean, np.std, np.max, np.min]:
            features.extend(stat(mfcc, axis=1))
        
        # Delta và Delta-Delta của MFCC
        for delta in [librosa.feature.delta(mfcc), librosa.feature.delta(mfcc, order=2)]:
            features.extend(np.mean(delta, axis=1))
            features.extend(np.std(delta, axis=1))
        
        # Spectral features
        for f in [librosa.feature.spectral_centroid, librosa.feature.spectral_bandwidth,
                  librosa.feature.spectral_rolloff, librosa.feature.spectral_flatness]:
            feat = f(y=audio, sr=self.sr) if 'sr' in str(f.__code__.co_varnames) else f(y=audio)
            features.extend([np.mean(feat), np.std(feat)])
        
        # Spectral contrast, Chroma, Tonnetz
        for f in [librosa.feature.spectral_contrast(y=audio, sr=self.sr), 
                  librosa.feature.chroma_stft(y=audio, sr=self.sr),
                  librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=self.sr)]:
            features.extend(np.mean(f, axis=1))
            features.extend(np.std(f, axis=1))
        
        # Zero crossing rate và RMS
        zcr = librosa.feature.zero_crossing_rate(audio)
        rms = librosa.feature.rms(y=audio)
        features.extend([np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)])
        
        return np.array(features)
    
    def predict_instrument(self):
        # Dự đoán nhạc cụ sử dụng phương pháp đã chọn (CNN hoặc SVM).
        if not self.file_exists or self.audio_data is None:
            msb.showerror("Error", "No audio loaded. Please record or open a file first.")
            return
        
        method = self.prediction_method
        
        # Kiểm tra model có sẵn không
        if method == 'CNN':
            if self.cnn_model is None or self.cnn_label_encoder is None:
                msb.showerror("Error", "CNN model not loaded. Please check model files.")
                return
            self.predict_with_cnn()
        elif method == 'SVM':
            if self.svm_pipeline is None or self.svm_label_encoder is None:
                msb.showerror("Error", "SVM model not loaded. Please check model files.")
                return
            self.predict_with_svm()
        else:
            msb.showerror("Error", f"Unknown method: {method}")
    
    def predict_with_cnn(self):
        """
        Dự đoán nhạc cụ sử dụng CNN segment-based approach với aggregation.
        Giống như trong training notebook: cắt thành segments, predict từng segment,
        rồi average softmax để có kết quả cuối cùng.
        """
        try:
            # Kiểm tra độ dài audio (tối thiểu 0.5 giây)
            min_duration = 0.5
            if len(self.audio_data) / self.sample_rate < min_duration:
                raise ValueError(f"Audio quá ngắn. Cần tối thiểu {min_duration} giây, "
                               f"hiện tại: {len(self.audio_data)/self.sample_rate:.2f} giây")
            
            self.update_status("Extracting segments...")
            self.update()
            
            # Resample nếu cần
            audio = self.audio_data.copy()
            if self.sample_rate != self.sr:
                audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=self.sr)
            
            # Normalize audio về [-1, 1] nếu cần
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.max(np.abs(audio))
            
            # Extract segments với sliding window (giống như trong training)
            segments = self.extract_sliding_segments(audio)
            
            self.update_status(f"Predicting {len(segments)} segments...")
            self.update()
            
            mel_segments = [self.segment_to_mel(seg) for seg in segments]
            
            # Chuẩn bị input cho model
            mel_segments_array = np.array(mel_segments)
            mel_segments_input = self.prepare_segments_input(mel_segments_array)
            
            # Predict với batch_size để tận dụng GPU/CPU tốt hơn
            batch_size = min(32, len(mel_segments_input))  # Batch size tối ưu
            segment_probs = self.cnn_model.predict(mel_segments_input, verbose=0, batch_size=batch_size)
            
            # Average softmax across all segments (segment aggregation)
            avg_probs = np.mean(segment_probs, axis=0)
            
            # Lấy top-3 predictions
            top3_indices = np.argsort(avg_probs)[::-1][:3]
            
            # Decode labels
            predicted_class_idx = top3_indices[0]
            predicted_class_code = self.cnn_label_encoder.inverse_transform([predicted_class_idx])[0]
            predicted_class_name = self.instrument_names.get(predicted_class_code, predicted_class_code)
            confidence = avg_probs[predicted_class_idx] * 100
            
            # Hiển thị kết quả
            self.lbl_result.config(
                text=f"Predicted: {predicted_class_name}",
                fg='darkgreen'
            )
            self.lbl_confidence.config(
                text=f"Confidence: {confidence:.2f}% ({len(segments)} segments) [CNN]"
            )
            
            # Hiển thị top-3
            top3_text = ""
            for i, idx in enumerate(top3_indices):
                class_code = self.cnn_label_encoder.inverse_transform([idx])[0]
                class_name = self.instrument_names.get(class_code, class_code)
                prob = avg_probs[idx] * 100
                top3_text += f"{i+1}. {class_name}: {prob:.2f}%\n"
            
            self.lbl_top3.config(text=top3_text)
            
            self.update_status("Prediction completed!")
            
            # Hiển thị message box với kết quả
            result_msg = f"Method: CNN (Segment-based)\n"
            result_msg += f"Predicted Instrument: {predicted_class_name}\n"
            result_msg += f"Confidence: {confidence:.2f}%\n"
            result_msg += f"Segments used: {len(segments)}\n\n"
            result_msg += "Top 3 Predictions:\n"
            for i, idx in enumerate(top3_indices):
                class_code = self.cnn_label_encoder.inverse_transform([idx])[0]
                class_name = self.instrument_names.get(class_code, class_code)
                prob = avg_probs[idx] * 100
                result_msg += f"{i+1}. {class_name}: {prob:.2f}%\n"
            
            msb.showinfo("Recognition Result", result_msg)
            
        except ValueError as e:
            msb.showerror("Error", str(e))
            self.update_status("Prediction failed: " + str(e))
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            msb.showerror("Error", error_msg)
            self.update_status("Prediction failed")
            print(f"Error in CNN prediction: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_with_svm(self):
        # Dự đoán nhạc cụ sử dụng SVM với handcrafted features.
        try:
            # Kiểm tra độ dài audio (tối thiểu 0.5 giây)
            min_duration = 0.5
            if len(self.audio_data) / self.sample_rate < min_duration:
                raise ValueError(f"Audio quá ngắn. Cần tối thiểu {min_duration} giây, "
                               f"hiện tại: {len(self.audio_data)/self.sample_rate:.2f} giây")
            
            self.update_status("Extracting handcrafted features...")
            self.update()
            
            # Resample nếu cần
            audio = self.audio_data.copy()
            if self.sample_rate != self.sr:
                audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=self.sr)
            
            # Normalize audio về [-1, 1] nếu cần
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.max(np.abs(audio))
            
            # Extract handcrafted features
            features = self.extract_handcrafted_features(audio)
            features = features.reshape(1, -1)
            
            self.update_status("Predicting with SVM...")
            self.update()
            
            # Preprocess với scaler và selector từ pipeline
            scaler = self.svm_pipeline['scaler']
            selector = self.svm_pipeline['selector']
            svm_model = self.svm_pipeline['model']
            
            features_scaled = scaler.transform(features)
            features_selected = selector.transform(features_scaled)
            
            # Predict
            predicted_class_idx = svm_model.predict(features_selected)[0]
            
            # Kiểm tra predicted_class_idx có hợp lệ không
            n_classes = len(self.svm_label_encoder.classes_)
            if predicted_class_idx < 0 or predicted_class_idx >= n_classes:
                raise ValueError(f"Invalid predicted class index: {predicted_class_idx}. "
                               f"Expected range: 0-{n_classes-1}. "
                               f"Model classes: {self.svm_label_encoder.classes_}")
            
            # Lấy decision scores và probabilities
            # Với OVO (one-vs-one), decision_function trả về scores cho từng cặp classes
            # Cần dùng predict_proba nếu có, hoặc tính từ decision_function
            try:
                # Thử dùng predict_proba nếu có
                probabilities = svm_model.predict_proba(features_selected)[0]
            except AttributeError:
                # Nếu không có predict_proba, dùng decision_function
                decision_scores = svm_model.decision_function(features_selected)
                
                # Với OVO, decision_function có shape (n_samples, n_classes * (n_classes-1) / 2)
                # Cần convert về probabilities cho n_classes
                if len(decision_scores[0]) > n_classes:
                    # OVO: tính vote cho mỗi class
                    # Mỗi cặp classes vote cho class có score cao hơn
                    votes = np.zeros(n_classes)
                    pair_idx = 0
                    for i in range(n_classes):
                        for j in range(i + 1, n_classes):
                            if decision_scores[0][pair_idx] > 0:
                                votes[i] += 1
                            else:
                                votes[j] += 1
                            pair_idx += 1
                    probabilities = votes / np.sum(votes) if np.sum(votes) > 0 else votes
                else:
                    # OVR hoặc binary: dùng trực tiếp
                    exp_scores = np.exp(decision_scores[0] - np.max(decision_scores[0]))
                    probabilities = exp_scores / np.sum(exp_scores) if np.sum(exp_scores) > 0 else exp_scores
            
            # Đảm bảo probabilities có đúng số classes
            if len(probabilities) != n_classes:
                raise ValueError(f"Probabilities length mismatch: {len(probabilities)} != {n_classes}")
            
            # Lấy top-3 predictions (đảm bảo indices hợp lệ)
            top3_indices = np.argsort(probabilities)[::-1][:3]
            top3_indices = [idx for idx in top3_indices if 0 <= idx < n_classes]
            
            # Decode labels với error handling
            try:
                predicted_class_code = self.svm_label_encoder.inverse_transform([predicted_class_idx])[0]
            except ValueError as e:
                raise ValueError(f"Label encoder error: {e}. "
                               f"Predicted index: {predicted_class_idx}, "
                               f"Available classes: {self.svm_label_encoder.classes_}")
            predicted_class_name = self.instrument_names.get(predicted_class_code, predicted_class_code)
            confidence = probabilities[predicted_class_idx] * 100
            
            # Hiển thị kết quả
            self.lbl_result.config(
                text=f"Predicted: {predicted_class_name}",
                fg='darkgreen'
            )
            self.lbl_confidence.config(
                text=f"Confidence: {confidence:.2f}% [SVM]"
            )
            
            # Hiển thị top-3
            top3_text = ""
            for i, idx in enumerate(top3_indices):
                try:
                    if 0 <= idx < n_classes:
                        class_code = self.svm_label_encoder.inverse_transform([idx])[0]
                        class_name = self.instrument_names.get(class_code, class_code)
                        prob = probabilities[idx] * 100
                        top3_text += f"{i+1}. {class_name}: {prob:.2f}%\n"
                except (ValueError, IndexError) as e:
                    # Bỏ qua nếu index không hợp lệ
                    continue
            
            self.lbl_top3.config(text=top3_text)
            
            self.update_status("Prediction completed!")
            
            # Hiển thị message box với kết quả
            result_msg = f"Method: SVM (Handcrafted Features)\n"
            result_msg += f"Predicted Instrument: {predicted_class_name}\n"
            result_msg += f"Confidence: {confidence:.2f}%\n\n"
            result_msg += "Top 3 Predictions:\n"
            for i, idx in enumerate(top3_indices):
                try:
                    if 0 <= idx < n_classes:
                        class_code = self.svm_label_encoder.inverse_transform([idx])[0]
                        class_name = self.instrument_names.get(class_code, class_code)
                        prob = probabilities[idx] * 100
                        result_msg += f"{i+1}. {class_name}: {prob:.2f}%\n"
                except (ValueError, IndexError) as e:
                    # Bỏ qua nếu index không hợp lệ
                    continue
            
            msb.showinfo("Recognition Result", result_msg)
            
        except ValueError as e:
            msb.showerror("Error", str(e))
            self.update_status("Prediction failed: " + str(e))
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            msb.showerror("Error", error_msg)
            self.update_status("Prediction failed")
            print(f"Error in SVM prediction: {e}")
            import traceback
            traceback.print_exc()
    
    def prepare_segments_input(self, mel_segments_array):
        """
        Chuẩn bị input cho CNN model từ array của mel spectrograms.
        Mỗi mel spectrogram sẽ được resize về đúng kích thước model expect.
        """
        if not hasattr(self, '_cached_input_shape'):
            model_input_shape = self.cnn_model.input_shape
            self._cached_input_shape = (model_input_shape[1], model_input_shape[2])
        
        target_height, target_width = self._cached_input_shape
        
        processed_segments = []
        for mel_spec in mel_segments_array:
            # Resize mel_spec về đúng kích thước
            current_height, current_width = mel_spec.shape
            zoom_factors = (target_height / current_height, target_width / current_width)
            mel_spec_resized = zoom(mel_spec, zoom_factors, order=1)
            
            # Đảm bảo đúng kích thước
            if mel_spec_resized.shape != (target_height, target_width):
                if mel_spec_resized.shape[0] > target_height:
                    mel_spec_resized = mel_spec_resized[:target_height, :]
                elif mel_spec_resized.shape[0] < target_height:
                    padding = np.zeros((target_height - mel_spec_resized.shape[0], mel_spec_resized.shape[1]))
                    mel_spec_resized = np.vstack([mel_spec_resized, padding])
                
                if mel_spec_resized.shape[1] > target_width:
                    mel_spec_resized = mel_spec_resized[:, :target_width]
                elif mel_spec_resized.shape[1] < target_width:
                    padding = np.zeros((mel_spec_resized.shape[0], target_width - mel_spec_resized.shape[1]))
                    mel_spec_resized = np.hstack([mel_spec_resized, padding])
            
            # Thêm channel dimension: (128, 130) -> (128, 130, 1)
            mel_spec_input = np.expand_dims(mel_spec_resized, axis=-1)
            processed_segments.append(mel_spec_input)
        
        # Stack thành batch: (num_segments, 128, 130, 1)
        return np.array(processed_segments)
    
    def update_status(self, message):
        # Cập nhật trạng thái
        self.lbl_status.config(text=message)
        self.update_idletasks()


if __name__ == "__main__":
    app = InstrumentRecognitionApp()
    app.mainloop()

