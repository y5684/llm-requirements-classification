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

We evaluate prompting-only inference with three prompt styles for each task (binary and 12-class). All prompts (i) use a concise system role, (ii) give strict final rules, and (iii) force an exact label line wrapped in <label>…</label> to enable deterministic parsing.

<details> <summary><code>binary / basic</code></summary>
system: >
  You are a precise software requirements engineer.
  Follow instructions exactly and respect output constraints.

user: |
  Decide whether the following requirement is a Functional Requirement (FR)
  or a Non-Functional Requirement (NFR).

  Requirement:
  {requirement}

  Final rules:
  - Output the final line exactly as one of:
    <label>FR</label>  or  <label>NFR</label>
  - Do not output anything after </label>.

  Answer:
</details>
