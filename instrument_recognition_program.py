"""
Musical Instrument Recognition Demo
Sử dụng CNN model đã train trên IRMAS dataset để nhận dạng nhạc cụ từ audio
Phương pháp: CNN Segment-based với Mel-Spectrogram
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
        self.playing = False
        self.playback_stopped = False
        self.file_exists = False
        self.audio_data = None
        self.sample_rate = None
        self.current_file = None
        
        # Load models và label encoders
        self.cnn_model = None
        self.cnn_label_encoder = None
        self.segment_config = None
        self.instrument_names = {}
        self.load_models()
        
        # Mel-spectrogram parameters (phải khớp với lúc training)
        self.sr = 22050  # Sample rate
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        
        # Segment-based parameters (load từ config hoặc default)
        if self.segment_config:
            self.segment_duration = self.segment_config.get('segment_duration', 2.0)
            self.segment_overlap = self.segment_config.get('segment_overlap', 0.5)
        else:
            self.segment_duration = 2.0  # Default từ training
            self.segment_overlap = 0.5
        self.segment_samples = int(self.sr * self.segment_duration)
        
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
        # Load CNN model từ thư mục IRMAS_Models
        try:
            # Load CNN model
            cnn_model_path = "IRMAS_Models/best_segment_cnn.keras"
            cnn_label_encoder_path = "IRMAS_Models/label_encoder_seg.joblib"
            
            if os.path.exists(cnn_model_path):
                self.cnn_model = keras.models.load_model(cnn_model_path)
                self.cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                print(f"CNN model loaded. Input shape: {self.cnn_model.input_shape}")
            else:
                msb.showerror("Error", f"CNN model not found: {cnn_model_path}")
                print(f"Warning: CNN model not found: {cnn_model_path}")
                return
            
            if os.path.exists(cnn_label_encoder_path):
                self.cnn_label_encoder = joblib.load(cnn_label_encoder_path)
                print(f"CNN label encoder loaded. Classes: {self.cnn_label_encoder.classes_}")
            else:
                msb.showerror("Error", f"CNN label encoder not found: {cnn_label_encoder_path}")
                print(f"Warning: CNN label encoder not found: {cnn_label_encoder_path}")
                return
            
            # Load segment config
            config_path = "IRMAS_Models/segment_config.joblib"
            if os.path.exists(config_path):
                self.segment_config = joblib.load(config_path)
                print(f"Segment config loaded: {self.segment_config}")
            
            # Kiểm tra xem model có được load không
            if self.cnn_model is None or self.cnn_label_encoder is None:
                msb.showerror("Error", "CNN model not loaded! Please check IRMAS_Models folder.")
                return
            
            msb.showinfo("Success", "CNN model loaded successfully!")
            
        except Exception as e:
            msb.showerror("Error", f"Failed to load models: {str(e)}")
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
    
    def setup_ui(self):
        # Thiết lập giao diện
        self.title('Musical Instrument Recognition Demo')
        self.geometry('1000x800')
        
        # Canvas để hiển thị waveform và spectrogram
        self.cvs_figure = tk.Canvas(self, width=700, height=500, relief=tk.SUNKEN, border=1, bg='white')
        
        # Tạo các LabelFrame
        lblf_controls = tk.LabelFrame(self, text="Controls", padx=5, pady=5)
        lblf_results = tk.LabelFrame(self, text="Recognition Results", padx=5, pady=5)
        lblf_status = tk.LabelFrame(self, text="Status", padx=5, pady=5)
        
        # Buttons trong controls frame - lưu reference để có thể enable/disable
        self.btn_open = tk.Button(lblf_controls, text='Open File', width=12, command=self.open_file)
        self.btn_record = tk.Button(lblf_controls, text='Record', width=12, 
                              command=lambda: self.threading_rec(1))
        self.btn_stop = tk.Button(lblf_controls, text='Stop', width=12, 
                            command=lambda: self.threading_rec(2),
                            state=tk.DISABLED)
        self.btn_play = tk.Button(lblf_controls, text='Play', width=12, 
                            command=lambda: self.threading_rec(3),
                            state=tk.DISABLED)
        self.btn_predict = tk.Button(lblf_controls, text='Predict', width=12, 
                               command=self.predict_instrument, 
                               bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                               state=tk.DISABLED)
        
        # Status label
        self.lbl_status = tk.Label(lblf_status, text="Ready", fg='green', 
                                   font=('Arial', 10, 'bold'))
        
        # File info label
        self.lbl_file_info = tk.Label(lblf_status, text="No file loaded", 
                                     font=('Arial', 8), fg='gray')
        
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
        
        # Layout controls
        self.btn_open.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.btn_record.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        self.btn_stop.grid(row=2, column=0, padx=5, pady=5, sticky='ew')
        self.btn_play.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
        self.btn_predict.grid(row=4, column=0, padx=5, pady=10, sticky='ew')
        lblf_controls.columnconfigure(0, weight=1)
        
        # Khởi tạo trạng thái nút
        self.update_buttons_state()
        
        # Layout status
        self.lbl_status.pack(pady=5)
        self.lbl_file_info.pack(pady=2)
        
        # Layout results
        self.lbl_result.pack(pady=5)
        self.lbl_confidence.pack(pady=2)
        self.lbl_top3_title.pack(pady=(10, 2))
        self.lbl_top3.pack(pady=2)
        
        # Main layout - sắp xếp theo hàng ngang
        self.cvs_figure.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        lblf_controls.grid(row=0, column=1, padx=5, pady=5, sticky='n')
        lblf_status.grid(row=1, column=0, padx=5, pady=2, sticky='ew', columnspan=2)
        lblf_results.grid(row=2, column=0, padx=5, pady=2, sticky='ew', columnspan=2)
        
        # Giới hạn chiều cao của results frame
        lblf_results.config(height=180)
        
        # Configure grid weights - cho phép mở rộng theo chiều ngang
        self.grid_rowconfigure(0, weight=0)  # Row 0 (canvas + controls) không mở rộng theo chiều dọc
        self.grid_rowconfigure(1, weight=0)  # Row 1 (status) không mở rộng
        self.grid_rowconfigure(2, weight=0)  # Row 2 (results) không mở rộng - cố định chiều cao
        self.grid_columnconfigure(0, weight=1)  # Column 0 (canvas) mở rộng theo chiều ngang
        self.grid_columnconfigure(1, weight=0)  # Column 1 (controls) không mở rộng
        
    def update_buttons_state(self):
        """
        Cập nhật trạng thái enable/disable của các nút dựa trên trạng thái hiện tại
        """
        # Khi đang record: chỉ enable Stop, disable tất cả nút khác
        if self.recording:
            self.btn_open.config(state=tk.DISABLED)
            self.btn_record.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_play.config(state=tk.DISABLED)
            self.btn_predict.config(state=tk.DISABLED)
        # Khi đang play: enable Stop, disable các nút khác
        elif self.playing:
            self.btn_open.config(state=tk.DISABLED)
            self.btn_record.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_play.config(state=tk.DISABLED)
            self.btn_predict.config(state=tk.DISABLED)
        # Khi không có file: chỉ enable Open và Record
        elif not self.file_exists:
            self.btn_open.config(state=tk.NORMAL)
            self.btn_record.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_play.config(state=tk.DISABLED)
            self.btn_predict.config(state=tk.DISABLED)
        # Khi có file và không đang làm gì: enable tất cả nút cần thiết
        else:
            self.btn_open.config(state=tk.NORMAL)
            self.btn_record.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_play.config(state=tk.NORMAL)
            self.btn_predict.config(state=tk.NORMAL)
    
    def callback(self, indata, frames, time, status):
        # Callback function cho sounddevice recording
        self.q.put(indata.copy())
    
    def threading_rec(self, mode):
        # Xử lý recording/stop/play trong thread riêng
        if mode == 1:  # Record
            if self.recording or self.playing:
                return  # Nút đã bị disable, không cần check nữa
            t1 = threading.Thread(target=self.record_audio)
            t1.daemon = True
            t1.start()
        elif mode == 2:  # Stop
            if self.recording:
                self.recording = False
                self.update_status("Recording stopped")
                self.update_buttons_state()
            elif self.playing:
                self.stop_playback()
            else:
                self.update_status("Nothing to stop")
        elif mode == 3:  # Play
            if self.recording or self.playing:
                return  # Nút đã bị disable, không cần check nữa
            self.play_audio()
    
    def record_audio(self):
        # Thu âm từ microphone
        self.recording = True
        self.update_buttons_state()
        self.update_status("Recording... Speak into the microphone")
        self.current_file = "recorded_audio.wav"
        
        try:
            with sf.SoundFile(self.current_file, mode='w', samplerate=self.sr, 
                            channels=1, subtype='PCM_16') as file:
                with sd.InputStream(samplerate=self.sr, channels=1, 
                                  callback=self.callback):
                    while self.recording:
                        try:
                            # Timeout để tránh block vô hạn nếu có lỗi
                            file.write(self.q.get(timeout=1.0))
                        except queue.Empty:
                            # Nếu queue rỗng, tiếp tục loop
                            continue
            
            # Load audio sau khi ghi xong
            self.audio_data, self.sample_rate = sf.read(self.current_file, dtype='float32')
            self.recording = False
            self.file_exists = True
            self.update_buttons_state()
            self.update_file_info()
            self.update_status("Recording finished. Ready to predict.")
            self.draw_waveform()
            msb.showinfo("Recording", "Recording finished successfully!")
            
        except Exception as e:
            msb.showerror("Error", f"Recording failed: {str(e)}")
            self.update_status("Recording failed")
            self.recording = False
            self.update_buttons_state()
    
    def open_file(self):
        # Mở file audio từ disk
        filetypes = (
            ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a"),
            ("Wave files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("FLAC files", "*.flac"),
            ("All files", "*.*")
        )
        filename = fd.askopenfilename(title="Open audio file", filetypes=filetypes)
        
        if filename:
            try:
                self.current_file = filename
                # Load audio với librosa để đảm bảo đúng format
                self.audio_data, self.sample_rate = librosa.load(filename, sr=None)
                self.file_exists = True
                self.update_buttons_state()
                self.update_file_info()
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
            self.playing = True
            self.playback_stopped = False
            self.update_buttons_state()
            self.update_status("Playing audio...")
            
            # Phát audio trong thread riêng để không block UI
            def play_thread():
                try:
                    sd.play(self.audio_data, self.sample_rate)
                    # Bắt đầu check playback status
                    self.after(100, check_playback)
                except Exception as e:
                    self.after(0, lambda: msb.showerror("Error", f"Playback failed: {str(e)}"))
                    self.after(0, lambda: self.update_status("Playback failed"))
                    self.playing = False
                    self.after(0, lambda: self.update_buttons_state())
            
            def check_playback():
                try:
                    if self.playback_stopped:
                        try:
                            sd.stop()
                        except:
                            pass
                        self.playing = False
                        self.playback_stopped = False
                        self.update_buttons_state()
                        self.update_status("Playback stopped")
                    elif self.playing:
                        # Kiểm tra xem stream còn active không
                        try:
                            stream = sd.get_stream()
                            if stream is None or not stream.active:
                                # Playback đã kết thúc
                                self.playing = False
                                self.update_buttons_state()
                                self.update_status("Playback finished")
                            else:
                                # Tiếp tục check
                                self.after(100, check_playback)
                        except:
                            # Nếu không check được, giả sử đã kết thúc
                            self.playing = False
                            self.update_buttons_state()
                            self.update_status("Playback finished")
                except Exception as e:
                    # Nếu có lỗi, dừng playback
                    self.playing = False
                    self.update_buttons_state()
                    self.update_status("Playback error")
            
            t = threading.Thread(target=play_thread)
            t.daemon = True
            t.start()
            
        except Exception as e:
            msb.showerror("Error", f"Playback failed: {str(e)}")
            self.update_status("Playback failed")
            self.playing = False
            self.update_buttons_state()
    
    def stop_playback(self):
        # Dừng playback hoàn toàn
        self.playback_stopped = True
        self.playing = False
        try:
            sd.stop()
        except:
            pass
        self.update_buttons_state()
        self.update_status("Playback stopped")
    
    def draw_waveform(self):
        # Vẽ waveform và spectrogram trên canvas
        if self.audio_data is None:
            return
        
        self.update_status("Drawing waveform and spectrogram...")
        self.update_idletasks()
        
        self.cvs_figure.delete(tk.ALL)
        width = self.cvs_figure.winfo_width()
        height = self.cvs_figure.winfo_height()
        
        if width <= 1 or height <= 1:
            width = 700
            height = 500
        
        # Tính duration của audio
        duration = len(self.audio_data) / (self.sample_rate if self.sample_rate else self.sr)
        
        # Chia canvas thành 2 phần: waveform (trên) và spectrogram (dưới)
        # Dành không gian cho time axis
        time_axis_height = 25
        waveform_height = (height - time_axis_height) // 2
        spectrogram_height = (height - time_axis_height) // 2
        
        # WAVEFORM
        self.update_status("Rendering waveform...")
        self.update_idletasks()
        
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
        separator_y = waveform_height
        self.cvs_figure.create_line(0, separator_y, width, separator_y, fill='gray', width=2)
        
        # SPECTROGRAM
        try:
            self.update_status("Rendering spectrogram...")
            self.update_idletasks()
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
            
            # Vẽ time axis chung ở dưới cùng
            time_axis_y = height - time_axis_height
            self.cvs_figure.create_line(0, time_axis_y, width, time_axis_y, fill='black', width=2)
            
            # Vẽ time labels (chia thành nhiều mốc)
            num_ticks = 20
            for i in range(num_ticks + 1):
                x_pos = int(i * width / num_ticks)
                time_val = i * duration / num_ticks
                
                # Format thời gian
                if duration < 1:
                    time_str = f"{time_val*1000:.0f}ms"
                elif duration < 60:
                    time_str = f"{time_val:.2f}s"
                else:
                    minutes = int(time_val // 60)
                    seconds = int(time_val % 60)
                    time_str = f"{minutes}:{seconds:02d}"
                
                # Tick mark (vẽ lên trên)
                tick_length = 8
                self.cvs_figure.create_line(x_pos, time_axis_y, x_pos, time_axis_y - tick_length, fill='black', width=1)
                
                # Label
                self.cvs_figure.create_text(x_pos, time_axis_y + 5, text=time_str, 
                                           font=('Arial', 8), fill='black', anchor='n')
            
            self.update_status("Rendering complete")
        except Exception as e:
            print(f"Error drawing spectrogram: {e}")
            import traceback
            traceback.print_exc()
            self.update_status("Rendering failed")
    
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
    
    def predict_instrument(self):
        # Dự đoán nhạc cụ sử dụng CNN
        if not self.file_exists or self.audio_data is None:
            msb.showerror("Error", "No audio loaded. Please record or open a file first.")
            return
        
        # Kiểm tra model có sẵn không
        if self.cnn_model is None or self.cnn_label_encoder is None:
            msb.showerror("Error", "CNN model not loaded. Please check model files.")
            return
        
        self.predict_with_cnn()
    
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
                text=f"Confidence: {confidence:.2f}% ({len(segments)} segments)"
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
            result_msg = f"Predicted Instrument: {predicted_class_name}\n"
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
    
    def update_file_info(self):
        # Cập nhật thông tin file (duration, sample rate)
        if self.audio_data is not None and self.sample_rate is not None:
            duration = len(self.audio_data) / self.sample_rate
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            milliseconds = int((duration % 1) * 1000)
            
            if minutes > 0:
                duration_str = f"{minutes}:{seconds:02d}.{milliseconds:03d}"
            else:
                duration_str = f"{seconds}.{milliseconds:03d}s"
            
            file_name = os.path.basename(self.current_file) if self.current_file else "recorded_audio.wav"
            info_text = f"File: {file_name} | Duration: {duration_str} | Sample Rate: {self.sample_rate} Hz"
            self.lbl_file_info.config(text=info_text, fg='darkblue')
        else:
            self.lbl_file_info.config(text="No file loaded", fg='gray')
    
    def update_status(self, message):
        # Cập nhật trạng thái
        self.lbl_status.config(text=message)
        
        message_lower = message.lower()
        if any(keyword in message_lower for keyword in ['failed', 'error', 'lỗi']):
            color = 'red'  # Lỗi - màu đỏ
        elif any(keyword in message_lower for keyword in ['completed', 'finished', 'ready', 'loaded', 'success']):
            color = 'green'  # Thành công - màu xanh lá
        elif any(keyword in message_lower for keyword in ['recording', 'playing', 'extracting', 'predicting', 'loading']):
            color = 'blue'  # Đang xử lý - màu xanh dương
        else:
            color = 'black'  # Mặc định - màu đen
        
        self.lbl_status.config(fg=color)
        self.update_idletasks()


if __name__ == "__main__":
    app = InstrumentRecognitionApp()
    app.mainloop()
