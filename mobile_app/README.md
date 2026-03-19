# Mobile App — Edge-AI Document Processing

CLI demo and integration guide for the on-device document processing pipeline.

## Quick Start

```bash
# Show model info and supported types
uv run python -m mobile_app.src.app info

# Generate sample images and process them (requires trained models)
uv run python -m mobile_app.src.app demo

# Process a single image
uv run python -m mobile_app.src.app process path/to/document.png

# Batch process a directory
uv run python -m mobile_app.src.app batch data/samples/arztbesuchsbestaetigung/ -o results.json
```

## CLI Commands

| Command   | Description                                      |
|-----------|--------------------------------------------------|
| `process` | Process a single document image (PNG/JPEG)       |
| `info`    | Show model names, sizes, and availability        |
| `batch`   | Process all images in a directory                |
| `demo`    | Generate sample images and run the full pipeline |

## Model Size Overview

| Model                        | Format    | Approx. Size |
|------------------------------|-----------|-------------|
| Classifier (EfficientNet-Lite0 INT8) | ONNX | ~5 MB   |
| Extractor Arztbesuch (DistilBERT INT8) | ONNX | ~20 MB |
| Extractor Reisekosten (DistilBERT INT8) | ONNX | ~20 MB |
| Extractor Lieferschein (DistilBERT INT8) | ONNX | ~20 MB |
| **Total**                    |           | **~65 MB**  |

## ONNX Runtime Mobile Integration

### Android (Java/Kotlin)

1. Add the ONNX Runtime dependency to `build.gradle`:
   ```gradle
   implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.17.0'
   ```
2. Place ONNX model files in `app/src/main/assets/models/`.
3. Load a session:
   ```kotlin
   val session = OrtEnvironment.getEnvironment().use { env ->
       env.createSession(
           readAssetAsBytes("models/classifier_int8.onnx"),
           OrtSession.SessionOptions()
       )
   }
   ```
4. Prepare input as a `float[1][3][224][224]` tensor, run the session, and read logits.

### iOS (Swift)

1. Add the ORT Swift package via SPM:
   ```
   https://github.com/microsoft/onnxruntime-swift-package-manager
   ```
2. Bundle models in the app target.
3. Load a session:
   ```swift
   let env = try OrtEnvironment(loggingLevel: .warning)
   let session = try OrtSession(env: env, modelPath: modelPath, sessionOptions: nil)
   ```
4. Create an `OrtValue` from the preprocessed image buffer, run the session, and decode output.

## API Reference — DocumentService

```python
from api.service import DocumentService

service = DocumentService(config_path="config.yaml")

# Process from bytes
result = service.process_image(image_bytes)

# Process from file
result = service.process_image_file("document.png")

# Supported types
types = service.get_supported_types()
# ["arztbesuchsbestaetigung", "reisekostenbeleg", "lieferschein"]

# Get JSON schema
schema = service.get_schema("arztbesuchsbestaetigung")
```

`ProcessingResult` fields:
- `document_type` — classified document type
- `fields` — extracted key-value pairs
- `confidence` — classification confidence (0–1)
- `raw_text` — OCR text (optional)
