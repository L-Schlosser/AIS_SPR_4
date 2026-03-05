# Demo Website Redesign — Dark AI Tech Theme

## Context
The current Streamlit demo website has correct technical content but looks plain and boring with default styling. It needs a professional, visually impressive redesign for presenting to university professors (grading) and BMD company stakeholders (business pitch). The goal is a dark AI tech aesthetic that matches the quality of the underlying engineering work.

## Approach
Full Streamlit redesign with aggressive custom CSS/HTML injection via `st.markdown(unsafe_allow_html=True)`. No new frameworks — stay within Streamlit + Plotly + custom CSS.

## Color Palette
- Background: `#0E1117` (Streamlit dark default) with `#1A1F2E` card backgrounds
- Primary accent: `#00D4FF` (electric cyan)
- Secondary accent: `#7C3AED` (purple)
- Success/accuracy: `#10B981` (green)
- Warning: `#F59E0B` (amber)
- Text: `#E2E8F0` (light gray), `#FFFFFF` (headers)
- Gradients: cyan→purple for hero elements, dark→darker for cards

## Shared Components (`demo/utils.py`)
- Full dark theme CSS override (hide Streamlit defaults, dark backgrounds, custom fonts)
- Gradient card component: `def metric_card(title, value, subtitle, color)`
- Styled section headers with accent underlines
- Dark sidebar with project branding
- Custom footer
- Plotly dark theme helper: `def dark_plotly_layout(fig)`
- All Plotly charts use `template="plotly_dark"` with custom accent colors

## Page 1: Home (`demo/Home.py`)
- Hero section: gradient background (`#0E1117` → `#1A1F2E`), large title "Edge-AI Document Processing", tagline "On-device classification and field extraction for BMD Go", accent stats
- 4 metric cards in a row: Document Types (3), AI Models (4), Accuracy (98.4%), Model Size (199 MB)
- "How It Works" — 5-step pipeline as styled numbered cards with icons and arrows
- Feature highlights row: Privacy (on-device), Speed (<2s), Accuracy (98.4%) — gradient-bordered cards
- Supported documents section: 3 cards (Arztbesuch, Reisekosten, Lieferschein) with field lists

## Page 2: Live Demo (`demo/pages/1_Live_Demo.py`)
- Dark-styled file uploader area
- Sample generation buttons as styled cards (one per doc type)
- Processing: step indicators showing pipeline progress
- Results in dark cards: classification result with confidence bar, extracted fields as key-value cards with colored labels
- Side-by-side columns: document image | extracted data
- OCR text in dark code block expander

## Page 3: Architecture (`demo/pages/2_Architecture.py`)
- Pipeline diagram as styled HTML cards connected by arrows (not GraphViz)
- Each pipeline stage: dark card with emoji icon, description, input→output
- Model specifications dark table (model name, architecture, format, size, quantization)
- Technology badges section: ONNX Runtime, DistilBERT, EfficientNet-Lite0, RapidOCR, Pydantic — each as a styled badge

## Page 4: Models & Metrics (`demo/pages/3_Models_and_Metrics.py`)
- Hero metric: 98.4% classification accuracy as large Plotly gauge (Indicator trace)
- Confusion matrix heatmap with dark color scheme (custom colorscale)
- Per-class precision/recall/F1 as grouped bar chart with accent colors
- NER section: micro F1 per document type as individual gauge charts
- Entity-level breakdown in dark-styled expanders with colored tables
- Model size comparison: horizontal bar chart showing each model's footprint

## Page 5: Tech Stack (`demo/pages/5_Tech_Stack.py`) — NEW PAGE
- Technology cards grid: each card has name, description, role in project
  - PyTorch → Training framework
  - ONNX Runtime → On-device inference engine
  - DistilBERT German → NER backbone (3 models)
  - EfficientNet-Lite0 → Classification backbone
  - RapidOCR → Optical character recognition
  - Pydantic → Data validation
  - Streamlit → Demo interface
- Training methodology section: two-phase transfer learning explanation, BIO tagging scheme
- Data pipeline: synthetic generation with Faker (German locale), 750 images + 4500 NER samples
- Quantization details: FP16 for classifier, INT8 for NER models
- Test coverage stats: 372 tests, 28 test files, unit/integration/e2e

## Page 6: About (`demo/pages/6_About.py`)
- Team member cards: name + role for all 5 members (Celina Binder, Natalie Eichsteininger, Klevi Hysenlli, Lorenz Schlosser, Raphael Suchomel)
- Project context: University student project for BMD Go mobile app
- Repository stats cards: 372 tests passing, 3 document types, 4 ONNX models, ~199 MB total
- License/disclaimer: research and educational purposes

## Additional Fixes
- Fix `.gitignore`: add `data/samples/**/*_label.json` pattern for untracked label files
- All existing functionality preserved (pipeline, processing, metrics loading)
