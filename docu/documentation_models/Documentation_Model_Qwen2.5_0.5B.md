# Model Documentation
**Baseline Model**

## 1. Model <!-- TODO: Model Name -->
| Field | Value |
|---|---|
| **Model Name** | _Qwen2.5-0.5B-Instruct_ |
| **Model Version** | _2.5_ |
| **Base Model** | _qwen2.5-0.5B-prefill-20e_ |


---
 
## 2. Motivation
 
> Why was this model selected for evaluation?
 
This model is extremely lightweight with only ~0.5B parameters, making it attractive for fast inference and low‑resource environments, as for instance a phone. It also offers instruction‑following capabilities, which are useful for cleaning and structuring OCR‑extracted text.

---
 
## 3. Architecture
 
### 3.1 Classification
 
The model is used as a lightweight LLM classifier. A short prompt is generated for each document, and the model returns a predicted class name. The output is normalized through keyword matching to map variations (e.g., “rechnung”, “arzt”) to the canonical class labels defined in the benchmark.
 
### 3.2 OCR

The model does not perform OCR itself. Instead it receives the synthetic text samples generated in the benchmark (like invoices, receipts, doctors notes) and classifies them. OCR extraction is handled separately in the overall pipeline. 

### 3.3 NET .MAUI compatibility 
The model cannot run directly inside a .NET MAUI application. It requires a Python backend or remote inference endpoint. The MAUI app would communicate with it via HTTP.

---
## 4. Results and Comparison to Baseline: Qwen2.5-0.5B
 
| Model          | Accuracy | Latency |
|----------------|---------------------|----------------|
| **This model= Baseline Model** | 39.0% |     1.13s  |
 

---
 
## 5. Strengths & Weaknesses
 
### Strengths
 
- latency is very low, making it suitable for real‑time classification
- models compact size enables fast loading and unloading
 
### Weaknesses
 
- accuracy is limited due to the small model size
- classification becomes unreliable for ambiguous or noisy text
 
---
 
## 6. Decision
 
**Decision:** 
The model was not selected as the final solution, as the accuracy was not sufficient for production‑level document classification. But it was selected as the Baseline for all the other models.

**Reasoning:**  
Although the latency is excellent and well below the one‑second threshold, the accuracy does not meet the required reliability. The model struggles with several document types, and the classification quality is not high enough to support a robust OCR pipeline.

---
 
## 7. Notes & References
- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
 
---
 
*Documented by: Celina Binder*