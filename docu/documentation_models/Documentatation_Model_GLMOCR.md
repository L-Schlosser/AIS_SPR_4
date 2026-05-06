# Model Documentation

## 1. Model <!-- TODO: Model Name -->
| Field | Value |
|---|---|
| **Model Name** | GLM-OCR |
| **Model Version** | glm-ocr:q8_0 |
| **Base Model** | GLM-V encoder–decoder architecture |


---
 
## 2. Motivation
 
> Why was this model selected for evaluation?
 
This OCR model has only 0.9b parameters, which seemed like a good starting point. It also reduced the cost and latency, making it suitable for edge deployments.
 
---
 
## 3. Architecture
 
### 3.1 Classification
The DocumentClassifier matches input text against predefined keyword lists for 7 document types (receipt, medical, etc.) to determine the best fit. Confidence is calculated as the highest keyword match count divided by total matches plus one, scaled by 1.5 and capped at 1.0. 

 
### 3.2 OCR

The LocalOCR class first initializes EasyOCR (English/German, GPU off) as primary, falling back to Tesseract if EasyOCR is unavailable. It extracts text via the selected engine's methods, returning raw text, line data, and the OCR engine used.

### 3.3 NET .MAUI compatibility 
The module is Python-based with no C# interop, so direct integration with .NET MAUI is impossible. It can only work indirectly via a REST API wrapper where the MAUI app sends HTTP requests to a Python server running the OCR logic.

---
## 4. Results and Comparison to Baseline: Qwen2.5-0.5B
 
| Model          | Accuracy | Latency |
|----------------|---------------------|----------------|
| **Baseline**   | 39%                   | 1.13              |
| **This model** | 92%       | 24.27 s  |
 

---
 
## 5. Strengths & Weaknesses
 
### Strengths
 
- accuracy is very strong, even for doctors notes the classification works well
 
### Weaknesses
 
- latency is high -> too high in order to be used in the apps
 
---
 
## 6. Decision
 
**Decision:**  
The integration of the model into the app was not done, as the latency was too high in order to get a nicely working app. 

**Reasoning:**
A latency of 2 seconds, as it is the case for this model here, is way above the threshold of maximum one second. Therefor the model integration was not pursued.
 
---
 
## 7. Notes & References
 
<!-- Optional: additional observations, links to papers, tickets, discussions -->
 
- https://docs.z.ai/guides/vlm/glm-ocr
 
---
 
*Documented by: Celina Binder*