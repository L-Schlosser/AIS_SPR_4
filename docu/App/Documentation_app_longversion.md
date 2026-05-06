# BMD Go Document Processor - Detailed Documentation


## Project Overview

The `bmd_go_document_processor` is a **Flutter-based mobile application** designed for the BMD Go project as part of the Edge-AI document processing system. This app serves as the **frontend interface** for capturing documents, performing on-device OCR, and classifying document types using locally-run ML models.

**Purpose**: This enable users to upload document types (medical confirmations, receipts, delivery notes) through a unified interface, with automated recognition and field extraction performed entirely on-device.


---

## Architecture & Data Flow

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│  Camera (multi-page) │ Gallery (multi-select) │ PDF Upload      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              DOCUMENT CAPTURE (lib/main.dart)                   │
│  • UploadScreen: main capture interface                         │
│  • _CameraMultiPageDialog: sequential photo capture             │
│  • Image preview with page management (add/remove/clear)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         OCR TEXT EXTRACTION (lib/services/ml_service_ocr.dart)  │
│  • Google ML Kit Text Recognition                               │
│  • PDF rendering via pdfx (converts PDF pages to images)        │
│  • Output: concatenated text string from all pages              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│    DOCUMENT CLASSIFICATION (lib/services/ml_service_classifier) │
│  • ONNX Runtime (onnxruntime package)                           │
│  • Model: assets/models/classifier_v2.onnx (to be added)        │
│  • Input: extracted text string                                 │
│  • Output: document type + confidence score                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│      RESULTS DISPLAY (lib/screens/classification_results)       │
│  • document type with color coding                              │
│  • confidence score (linear percent indicator)                  │
│  • extracted information display                                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **On-Device Processing**: no cloud dependencies for core functionality
2. **Multi-Page Support**: camera dialog allows sequential captures, the gallery supports multi-select
3. **PDF Support**: PDFs are rendered to images via `pdfx` before OCR, for better accuracy
4. **German UI**: all user-facing text is in German (Hochladen, Scannen, Verarbeiten, etc.)

---

## Project Structure

```
bmd_go_document_processor/
│
├── lib/                              # Dart source code
│   ├── main.dart                     # App entry, UploadScreen, Camera dialog
│   ├── screens/
│   │   └── classification_results_screen.dart   # Results display
│   └── services/
│       ├── image_picker_service.dart # Camera/gallery/file picking
│       ├── ml_service_classifier.dart# ONNX classification
│       └── ml_service_ocr.dart       # ML Kit OCR + PDF processing
│
├── test/                             # Unit and widget tests (empty/stubbed)
│
├── assets/                           
│   └── models/
│       ├── classifier_v2.onnx        # ONNX model (must be added manually)
│       └── doc_types.json            # Document type config (must be added manually)
│
├── android/                          # Android platform-specific code
│   ├── app/
│   ├── build.gradle
│   └── src/main/
│       ├── AndroidManifest.xml       # Permissions: camera, storage
│       └── kotlin/.../MainActivity.kt
│
│
├── pubspec.yaml                      # Dependencies & asset declarations
├── pubspec.lock                      # Auto-generated dependency lock file
├── analysis_options.yaml             # Linting rules
├── README.md                         # Default Flutter README
├── .metadata                        # Flutter project metadata
├── .gitignore                        # Git ignore rules
├── bmd_go_document_processor.iml     # IntelliJ project file
├── flutter_01.png                    # Screenshot/image asset
└── bugreport-*.zip                   # Crash report (should be in .gitignore)
```
---

## Source Code Analysis

### 1. `lib/main.dart`

**Classes:**
- `MyApp`: root widget - MaterialApp with orange theme
- `MainNavigationScreen`: stateful widget hosting the main structure
- `UploadScreen`: stateful widget - primary document capture interface
- `_CameraMultiPageDialog`: dialog widget for sequential camera capture
  
    
**UI Language**: German (`'BMD Dokumenterfassung'`, `'Belegupload'`, `'Scannen'`, `'Verarbeiten'`)

---

### 2. `lib/services/image_picker_service.dart`

**Methods:**
| Method | Description |
|--------|-------------|
| `takePhotoWithCamera()` | single photo capture with 95% quality, rear camera |
| `pickDocumentFile()` | file picker for jpg, jpeg, png, pdf, doc, docx |
| `pickMultipleImagesForDocument()` | multi-image selection from gallery only |

**Dependencies:** `image_picker`, `file_picker`

---

### 3. `lib/services/ml_service_ocr.dart`

**Classes:**
- `OCRClassificationResult`: data class holding `documentType`, `confidence`, `infos` map
- `OCRMLService`: Google ML Kit text recognition wrapper

**Methods:**
| Method | Description |
|--------|-------------|
| `processImages(List<File>)` | iterates images, runs ML Kit recognition, returns concatenated text |
| `processPdf(File)` | renders PDF pages to PNG images (2x resolution), then calls `processImages()` |
| `dispose()` | closes the text recognizer |

**PDF Processing Detail:**
```dart
// PDF pages are rendered at 2x resolution for better OCR accuracy
      final pageImage = await page.render(
        width: page.width * 2,
        height: page.height * 2,
        format: PdfPageImageFormat.png,
      );
// temporary files stored in system temp directory
```

---

### 4. `lib/services/ml_service_classifier.dart`

**Class:** `MLServiceClassifier`

**Methods:**
| Method | Description |
|--------|-------------|
| `initialize()` | loads ONNX model from `assets/models/classifier_v2.onnx` into `OrtSession` |
| `classify(String)` | runs inference: creates tensor from text, executes session, parses outputs |
| `dispose()` | releases ONNX session |

**Model Input/Output:**
- **Input**: text string as tensor with shape `[1, 1]`
- **Output 1**: document type label (string)
- **Output 2**: probability/confidence map (reduces to max value)

**Important**: The classifier expects the ONNX model to accept text input directly (not preprocessed embeddings).

---

### 5. `lib/screens/classification_results_screen.dart`

**Class:** `ClassificationResultsScreen` (Stateful)

**Display Elements:**
1. **Header**: colored container with document type (color-coded by type)
2. **Confidence Score**: linear percent indicator (green) - only shown if confidence > 0
3. **Information List**: key-value pairs from `result.infos`
4. **Action Buttons**: "Zurück" (back) and "Speichern" (save - not implemented!)

**Color Mapping:**
| Document Type | Color |
|--------------|-------|
| receipt | Orange |
| invoice | Blue |
| doctor_note | Red |
| care_leave | Purple |
| delivery_note | Green |
| master_data | Indigo |
| default | Grey |
---

## Dependencies

### Production Dependencies (`pubspec.yaml`)

**Flutter Core:**
```yaml
flutter:
  sdk: flutter
cupertino_icons: ^1.0.8
```

**Document Capture:**
```yaml
camera: ^0.10.5              # Camera access
image_picker: ^1.0.4         # Gallery selection
file_picker: ^8.1.2          # Document file picking (PDF, DOC, etc.)
flutter_pdfview: ^1.3.2      # PDF viewing (legacy)
pdfx: ^2.6.0                 # PDF rendering (used in code)
image: ^4.1.0                # Image processing
```

**Machine Learning:**
```yaml
onnxruntime: ^1.4.0                      # ONNX model inference
google_mlkit_text_recognition: ^0.11.0   # On-device OCR
# flutter_gemma: ^0.13.2                 # Commented out - Gemma API
# tflite_flutter: ^0.10.4                # Commented out - TF Lite
```

**State & Navigation:**
```yaml
provider: ^6.1.0              # State management
go_router: ^11.1.0            # Routing (not used in current code)
```

**Storage & UI:**
```yaml
sqflite: ^2.3.0               # Local database (not used yet)
path_provider: ^2.1.1         # File system paths
file: ^7.0.0                  # File operations
percent_indicator: ^4.1.0     # Confidence score display
```

### Asset Declarations

```yaml
flutter:
  assets:
    - assets/models/classifier_v2.onnx    
    - assets/models/doc_types.json        
  uses-material-design: true
```

---

## Platform Support

### Platform-Specific Folders

| Platform | Folder | Status | Notes |
|----------|--------|--------|-------|
| Android | `android/` | ✅ supported | requires camera/storage permissions |
| Web | `web/` | ✅ supported | limited by browser APIs |

---

## Integration with AIS_SPR_4
### Data Flow Classification Model

```
Python Training Scripts
       ↓
  ONNX Export
       ↓
[assets/models/classifier_v2.onnx]
       ↓
  Flutter App (ONNX Runtime)
       ↓
  Classification Result
```

---

## Build & Run Instructions

### Prerequisites

- **Flutter SDK**: ^3.11.1 (check with `flutter --version`)
- **Dart SDK**: included with Flutter
- **Android Studio** (for Android builds)
- **Xcode** (for iOS builds, macOS only)
- **Visual Studio 2022** (for Windows builds)

### Setup

```bash
# Navigate to the Flutter app
cd test_project/bmd_go_document_processor

# Install dependencies
flutter pub get

# Verify Flutter installation
flutter doctor
```

### Running the App

```bash
# Debug mode (default device)
flutter run

# Specific platforms
flutter run -d android    # Android device/emulator
flutter run -d chrome     # Web browser
```

### Building Release

```bash
# Android APK (release)
flutter build apk --release

# Web
flutter build web --release
```

### Adding Required Assets

```bash
# Create asset directories
mkdir assets
mkdir assets/models

# Copy your trained ONNX model
cp /path/to/classifier_v2.onnx assets/models/

# Create doc_types.json if needed
echo '{"types": ["receipt", "invoice", "doctor_note"]}' > assets/models/doc_types.json

# Update pubspec.yaml if you add new assets
# Then run:
flutter pub get
```

---
*Documented by: Celina Binder*