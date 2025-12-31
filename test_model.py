# Script test nhanh để kiểm tra Segment-based CNN model và label encoder có load được không

import numpy as np
from tensorflow import keras
import joblib
import os

def test_model():
    # Test Segment-based CNN model
    model_path = "IRMAS_Models/best_segment_cnn.keras"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return False
    
    try:
        print(f"Loading Segment-based CNN model from: {model_path}")
        model = keras.models.load_model(model_path)
        print(f"[OK] Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Number of parameters: {model.count_params():,}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False
    
    # Test label encoder cho segment-based CNN
    label_encoder_path = "IRMAS_Models/label_encoder_seg.joblib"
    if not os.path.exists(label_encoder_path):
        print(f"[ERROR] Label encoder not found: {label_encoder_path}")
        return False
    
    try:
        print(f"\nLoading label encoder from: {label_encoder_path}")
        label_encoder = joblib.load(label_encoder_path)
        print(f"[OK] Label encoder loaded successfully!")
        print(f"   Classes: {label_encoder.classes_}")
        print(f"   Number of classes: {len(label_encoder.classes_)}")
    except Exception as e:
        print(f"[ERROR] Failed to load label encoder: {e}")
        return False
    
    # Test segment config
    config_path = "IRMAS_Models/segment_config.joblib"
    if os.path.exists(config_path):
        try:
            print(f"\nLoading segment config from: {config_path}")
            config = joblib.load(config_path)
            print(f"[OK] Segment config loaded successfully!")
            print(f"   Segment duration: {config.get('segment_duration', 'N/A')}s")
            print(f"   Segment samples: {config.get('segment_samples', 'N/A')}")
            print(f"   Overlap: {config.get('segment_overlap', 'N/A')}")
        except Exception as e:
            print(f"[WARNING] Failed to load config: {e}")
    
    # Test prediction với dummy data
    print(f"\nTesting prediction with dummy data...")
    try:
        # Tạo dummy mel-spectrogram với shape (1, 128, 87, 1) - shape của segment-based CNN
        # - 128: số mel bins (N_MELS_SEG = 128)
        # - 87: số time frames = int(SEGMENT_SAMPLES / HOP_LENGTH_SEG) + 1
        #       = int(22050 * 2.0 / 512) + 1 = int(44100 / 512) + 1 = 86 + 1 = 87
        #       (Segment 2 giây với hop_length=512)
        # - 1: channel dimension cho CNN
        dummy_input = np.random.rand(1, 128, 87, 1).astype(np.float32)
        predictions = model.predict(dummy_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = predictions[0][predicted_class_idx] * 100
        
        print(f"[OK] Prediction test successful!")
        print(f"   Predicted class: {predicted_class}")
        print(f"   Confidence: {confidence:.2f}%")
        
        # Hiển thị tất cả probabilities
        print(f"\n   All probabilities:")
        all_probs = []
        for idx in range(len(predictions[0])):
            class_name = label_encoder.inverse_transform([idx])[0]
            prob = predictions[0][idx] * 100
            all_probs.append((class_name, prob))
        
        # Sắp xếp theo probability giảm dần
        all_probs.sort(key=lambda x: x[1], reverse=True)
        for class_name, prob in all_probs:
            print(f"      {class_name}: {prob:.2f}%")
    except Exception as e:
        print(f"[ERROR] Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("OK")
    return True

if __name__ == "__main__":
    test_model()

