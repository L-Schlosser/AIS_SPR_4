# Model Documentation

## 1. Model <!-- TODO: Model Name -->
| Field | Value |
|---|---|
| **Model Name** | SmolLM2-135M |
| **Model Version** | v2.0 |
| **Base Model** | SmolLM‑135M |


---
 
## 2. Motivation
 
> Why was this model selected for evaluation?

The model is small and has low latency between outputs while also being and LLM minimodel which can be used for both the classification and the NLP task. It doesn't take a lot of memory since it only has 135 milion parameters which makes it quicker to load.
 
<!-- TODO: 2–5 sentences describing the rationale -->
 
---
 
## 3. Architecture
 
### 3.1 Classification
 
<!-- TODO: Describe the classification architecture in plain language -->
Mini LLM which can be used to summarize and classify the output by prompt enginnering rather than fine-tuning.
 
### 3.2 OCR

<!-- TODO: Describe the classification architecture in plain language --> 

It can summarize the OCR output since it is very good at NLP (natural language processing) tasks.

### 3.3 NET .MAUI compatibility 
It is compatable but not natively. The model would need to be imported and then run via ONNX Runtime Mobile on the backend.

---
## 4. Results
 
Results and Comparison to Baseline: Qwen2.5-0.5B
| Model          | Accuracy | Latency |
|----------------|---------------------|----------------|
| Baseline   | 39.0%                   | 1.13s              |
| This model | <!-- TODO --> 16.0%      | <!-- TODO -->0.35s  |
---
 

## 6. Strengths & Weaknesses
 
### Strengths

- It is easy to load since the model is part of huggingface.
- It takes little time to be stored on the storage since it is small. 
- It has decent accuracy and latency response for it's size.
 
### Weaknesses
 
- <!-- TODO -->Doesn't have built in OCR so we would need an OCR if we would want to summarize the findings.
- The accuracy and classifications are not the best to use for deployed products that affect thousands of people.
---
 
## 7. Decision
 
**Decision:** 
We decided to reject the model. 
 
**Reasoning:**
The results were below the required threshold although the latency was within the threshold
 
<!-- TODO: 2–4 sentences explaining the decision -->
---
 
## 8. Notes & References
 
<!-- Optional: additional observations, links to papers, tickets, discussions -->
- HuggingFaceTB/SmolLM2-135M
 
---
 
*Documented by: Klevi Hysenlli*