from pathlib import Path
from tensorflow.keras.models import load_model

MODEL_PATH = Path("model.h5")

print("Checking model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Could not find {MODEL_PATH.resolve()}"
    )

model = load_model(MODEL_PATH)

print("Model loaded successfully!")
print(f"Input shape : {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Parameters  : {model.count_params():,}")