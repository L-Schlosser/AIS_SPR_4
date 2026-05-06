# Model Documentation

## 1. Model <!-- TODO: Model Name -->
| Field | Value |
|---|---|
| **Model Name** | DeBERTa-v3-XSmall |
| **Model Version** | v3.0-XSmall |
| **Base Model** | DeBERTa‑v1 |


---
 
## 2. Motivation
 
> Why was this model selected for evaluation?

The model is very small and has very low latency between outputs.
 It is an NLP model which can be used for the summarization task. It doesn't take a lot of memory since it has roughly 70 milion parameters which makes it very quick.
 
<!-- TODO: 2–5 sentences describing the rationale -->
 
---
 
## 3. Architecture
 
### 3.1 Classification
 
<!-- TODO: Describe the classification architecture in plain language -->
NLP model which can be used to summarize the output.
It can also be fine-tuned directly and trained accordingly to the data provided.
 
### 3.2 OCR

<!-- TODO: Describe the classification architecture in plain language --> 

It can summarize the OCR output since it is an NLP model( natural language processing ) tasks.

### 3.3 NET .MAUI compatibility 
It is compatable but not Natively. You would need to expose the inference via a REST API and then
Call the API from the MAUI app using HttpClient.

---
## 4. Results
 
Results and Comparison to Baseline: Qwen2.5-0.5B
| Model          | Accuracy | Latency |
|----------------|---------------------|----------------|
| Baseline   | 39.0%                   | 1.13s              |
| This model | <!-- TODO --> 15.0%     | <!-- TODO -->0.14s  |
---
 

## 6. Strengths & Weaknesses
 
### Strengths

- It is easy to load since the model is part of huggingface.
 -  It takes little time to be stored on the storage since it is small. 
- It has decent accuracy and latency response for it's size.
 
### Weaknesses
 
- <!-- TODO -->Doesn't have built in OCR so we would need an OCR if we would want to summarize the findings.
- The accuracy and classifications are not the best to use for deployed products that affect thousands of people.
- We would need to apply another classification model to perform the classification task.
- Not able to perform multi-tasks, only NLP summarization.
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
- microsoft/deberta-v3-small
 
---
 
*Documented by: Klevi Hysenlli*