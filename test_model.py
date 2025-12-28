# Script test nhanh để kiểm tra model và label encoder có load được không

import numpy as np
from tensorflow import keras
import joblib
import os

def test_model():
    # Test model
    model_path = "IRMAS_Models/best_cnn_model.keras"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return False
    
    try:
        print(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        print(f"[OK] Model loaded successfully!")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Number of parameters: {model.count_params():,}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False
    
    # Test label encoder
    label_encoder_path = "IRMAS_Models/label_encoder.joblib"
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
    
    # Test prediction với dummy data
    print(f"\nTesting prediction with dummy data...")
    try:
        # Tạo dummy mel-spectrogram với shape (1, 128, 130, 1)
        dummy_input = np.random.rand(1, 128, 130, 1).astype(np.float32)
        predictions = model.predict(dummy_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = predictions[0][predicted_class_idx] * 100
        
        print(f"[OK] Prediction test successful!")
        print(f"   Predicted class: {predicted_class}")
        print(f"   Confidence: {confidence:.2f}%")
        print(f"   All probabilities: {predictions[0]}")
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

