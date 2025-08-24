# results-llm-requirements-classification
Complete experimental results for the paper “Requirement Classification with Large Language Models: Empirical Insights into Fine-Tuning and Prompting Techniques”: binary &amp; 12-class metrics.

# Results — LLMs for Requirements Classification
Complete experimental results for the paper **“Requirement Classification with Large Language Models: Empirical Insights into Fine-Tuning and Prompting Techniques.”**  
Tasks: **Binary (FR vs. NFR)** and **12-Class (FR + 11 NFR subcategories)**.  
Experiments: **Tuning on non-instruction checkpoints** and **Prompting-only on instruction-tuned models**.
**Full metrics tables**: Accuracy, Weighted-Precision, Weighted-Recall and Weighted-F1 for RQ1–RQ3.

---

# RQ1 & RQ2
Full results for fine-tuning methods (RQ1) and for model size/architecture comparisons (RQ2) are summarized in the three tables below.
## 📊 Encoder-only
![RQ1 Leaderboard](results/Sheet1.png)

## 📊 Encoder-Decoder
![RQ1 Leaderboard](results/Sheet2.png)

## 📊 Decoder-only
![RQ1 Leaderboard](results/Sheet3.png)

---

# RQ3
We evaluate **six prompting templates** (2 tasks × 3 styles). All templates enforce an exact final label in `<label>…</label>`.

**Listing 1. Binary — Basic**
```text
System: You are a precise software requirements engineer. Follow instructions exactly and respect output constraints.
User: Decide whether the following requirement is a Functional Requirement (FR)
or a Non-Functional Requirement (NFR).

Requirement:
{requirement}

Final rules:
- Output the final line exactly as one of:
  <label>FR</label>  or  <label>NFR</label>
- Do not output anything after </label>.

Answer:
```
**Listing 1. Binary — Explain**

