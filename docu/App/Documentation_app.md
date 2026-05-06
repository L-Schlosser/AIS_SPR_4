# BMD Go Document Processor - Flutter Mobile App

## Overview

The `bmd_go_document_processor` is a Flutter-based mobile application designed for the BMD Go project, implementing Edge-AI document processing directly on mobile devices. This app serves as the primary user interface for document capture, classification, and information extraction.

---

## Architecture

### High-Level Flow

```
User Input (Camera/Gallery/PDF)
    ↓
Document Capture (multi-page support)
    ↓
OCR Text Extraction (Google ML Kit)
    ↓
Document Classification (ONNX Runtime)
    ↓
Results Display (JSON output)
```

### Key Components

#### 1. Entry Point - `lib/main.dart`
- **MainNavigationScreen**: root widget with app bar and navigation
- **UploadScreen**: primary screen for document capture
  - camera capture with multi-page support
  - gallery image selection (multiple images)
  - PDF document import
  - page management (remove/clear)
  - processing pipeline trigger

#### 2. Services Layer (`lib/services/`)

| File | Purpose |
|------|---------|
| `image_picker_service.dart` | handles camera, gallery, and document file picking with multi-page support |
| `ml_service_classifier.dart` | ONNX runtime integration for document type classification |
| `ml_service_ocr.dart` | Google ML Kit text recognition for OCR |

#### 3. Screens (`lib/screens/`)

| File | Purpose |
|------|---------|
| `classification_results_screen.dart` | displays classification results and extracted information |

---

## Dependencies (pubspec.yaml)

### Core Flutter
- `flutter` - SDK
- `cupertino_icons` - iOS-style icons

### Document Processing
- `camera` - camera access
- `image_picker` - image selection from gallery
- `file_picker` - PDF/document file selection
- `flutter_pdfview`, `pdfx` - PDF rendering

### Machine Learning
- `onnxruntime` - ONNX model inference engine
- `google_mlkit_text_recognition` - On-device OCR

### Utilities
- `image` - image processing
- `path_provider` - file system paths
- `file` - file operations
- `provider` - state management
- `go_router` - navigation
- `sqflite` - local database
- `percent_indicator` - progress indicators

---

## Platform Support

The app supports multiple platforms via platform-specific folders:

| Platform | Folder | Status |
|----------|--------|--------|
| Android | `android/` | supported |
| iOS | `ios/` | supported |
| Windows | `windows/` | supported (desktop testing) |
| Linux | `linux/` | supported |
| macOS | `macos/` | supported |
| Web | `web/` | supported |

### Windows-Specific Notes
The `windows/` folder contains:
- `flutter/generated_plugin_registrant.cc` - auto-generated plugin registration
- native Windows build configuration
- enables desktop testing of the mobile app logic

---

## Key Features

### 1. Multi-Page Document Capture
- capture multiple photos sequentially with camera
- select multiple images from gallery
- import multi-page PDFs
- preview thumbnails in grid layout
- remove individual pages or clear all

### 2. On-Device ML Inference
- **Classification**: ONNX Runtime executes `classifier_v2.onnx` locally
- **OCR**: Google ML Kit extracts text without server calls
- no internet required after model loading

### 3. Document Types
The app classifies documents into categories (defined in `doc_types.json`):
e.g.:
- Medical visit confirmations
- Business travel expense receipts
- Delivery and logistics documents

### 4. Results Display
- document type prediction
- confidence scores
- extracted text preview
- structured field output

---

## Development Setup

### Prerequisites
- Flutter SDK (^3.11.1)
- Dart SDK
- Android Studio / Xcode (for mobile deployment)
- Visual Studio (for Windows build)

### Installation
```bash
cd test_project/bmd_go_document_processor
flutter pub get
```

### Running the App
```bash
# Debug mode
flutter run

# Windows desktop
flutter run -d windows

# Android
flutter run -d android

# iOS (macOS only)
flutter run -d ios
```

### Building
```bash
# Android APK
flutter build apk

# iOS (macOS only)
flutter build ios

# Windows
flutter build windows
```

---

## Integration with Main Project

### Model Export Pipeline
1. Python backend (`app/`, `ocr/`) trains models
2. Models exported to ONNX format
3. Place ONNX files in `assets/models/`
4. Flutter app loads models via ONNX Runtime

### Data Flow
```
Python Training → ONNX Export → Flutter Asset → ONNX Runtime → Classification
```

---

## Testing

```bash
# Run Flutter tests
flutter test

# Run with coverage
flutter test --coverage
```

Test files located in `test/` directory.

---

## Project Structure

```
bmd_go_document_processor/
├── lib/
│   ├── main.dart                          # App entry point & upload screen
│   ├── screens/
│   │   └── classification_results_screen.dart  # Results display
│   └── services/
│       ├── image_picker_service.dart      # Camera/gallery/file picking
│       ├── ml_service_classifier.dart     # ONNX classification
│       └── ml_service_ocr.dart            # ML Kit OCR
├── assets/
│   └── models/
│       ├── classifier_v2.onnx             # Classification model
│       └── doc_types.json                 # Document types config
├── test/                                  # Unit & widget tests
├── android/                               # Android platform code
├── ios/                                   # iOS platform code
├── windows/                               # Windows platform code
├── linux/                                 # Linux platform code
├── macos/                                 # macOS platform code
├── web/                                   # Web platform code
├── pubspec.yaml                           # Dependencies & assets
└── README.md                              # Flutter starter README
```

---
*Documented by: Celina Binder*