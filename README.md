# 🎲 LLM Bias Robustness: A Monte Carlo Simulation Framework

> **Mila – Quebec AI Institute | HEC Montréal — MATH 60603**  
> Yanis Bencheikh — Automne 2025

---

## 📌 Overview

Standard LLM bias benchmarks evaluate stereotype preference on **clean, unperturbed text**. But how stable are these scores when the *surface* of sentences changes slightly — punctuation, synonyms, stopwords — without any change in meaning?

This project proposes a **systematic Monte Carlo simulation framework** to measure the **robustness of stereotype bias scores** (Stereo Win Rate) across 4 open-weight LLMs (~7–8B parameters), 3 datasets (EN + FR), and 4 levels of linguistic noise.

> **Core finding:** Bias scores measured on clean data are systematically *optimistic*. All models show significant, monotonically increasing non-robustness as noise grows — meaning reported bias figures are sensitive to surface-level text perturbations that carry no semantic change.

---

## 🧪 Models Evaluated

| Model | Architecture |
|---|---|
| **BLOOMz-7B1-MT** | Multilingual instruction-tuned |
| **Gemma-7B-IT** | Google, instruction-tuned |
| **Llama-3.1-8B-Instruct** | Meta, instruction-tuned |
| **Mistral-7B-Instruct-v0.3** | Mistral AI, instruction-tuned |

All models loaded in **4-bit quantization (BitsAndBytes)** on A100 GPU for memory-efficient inference. Log-probabilities extracted token-by-token for SWR computation.

---

## 📦 Datasets

| Dataset | Language | Bias Categories | Format |
|---|---|---|---|
| **CrowS-Pairs** (EN) | English | Gender, race, religion, sexual orientation | Minimal pairs |
| **StereoSet** (EN, intrasentence) | English | Multi-domain stereotypes | Context + completion |
| **CrowS-Pairs** (FR) | French | Same as EN (translated + adapted) | Minimal pairs |

Each dataset is converted to a unified triplet format: **(context xᵢ, anti-stereotyped completion yᵢᵃⁿᵗⁱ, stereotyped completion yᵢˢᵗᵉʳᵉᵒ)**.

---

## 📐 Methodology

### Stereo Win Rate (SWR)

The primary metric — proportion of pairs where the model assigns higher log-probability to the stereotyped completion:

```
SWR = (1/N) Σᵢ 𝟙[log p_θ(yᵢˢᵗᵉʳᵉᵒ | xᵢ) > log p_θ(yᵢᵃⁿᵗⁱ | xᵢ)]
```

SWR = 50% → no bias. SWR > 50% → preference for stereotyped completions.

### Noise Perturbation Schema (English — 4 Cumulative Doses)

| Dose | Perturbations Applied |
|---|---|
| **Dose 0** | Baseline (clean data) |
| **Dose 1** | Punctuation edits (add/remove commas, spaces, permute symbols) |
| **Dose 2** | Dose 1 + random stopword insertions (`"the"`, `"and"`, syntactically neutral) |
| **Dose 3** | Dose 2 + contraction expansion (`"isn't"` → `"is not"`, `"I'm"` → `"I am"`) |
| **Dose 4** | Dose 3 + synonym substitution for non-bias-bearing content words |

**French noise** follows the same logic (punctuation + stopwords + limited synonyms), with conservative synonym scope to avoid semantic drift.

### Bias-Word Protection

For each sentence pair, words that appear in one completion but not the other (the **bias-bearing tokens**) are identified and **excluded from all perturbations**. This ensures that:
- The stereotyped/anti-stereotyped polarity is never inverted
- SWR changes reflect model sensitivity to surface noise, not semantic change

### Monte Carlo Protocol

```
For each (model m, dataset d, dose δ):
    Compute SWR₀(m, d)  ← baseline on clean data
    For r = 1 to R:
        Generate noised dataset dᵟ,ᵣ  ← same random seed across models
        Compute SWRᵟ,ᵣ(m, d)
    
    μ_rep = mean(SWRᵟ,₁..ᵣ)
    σ_rep = std(SWRᵟ,₁..ᵣ)
    Z = (SWR₀ - μ_rep) / σ_rep     ← non-robustness Z-score
    p = 2 * (1 - Φ(|Z|))            ← two-sided p-value
```

Shared random seeds across models allow **conditional comparisons** — every model sees the exact same noised texts per replica.

### Statistical Outputs

- **Z-score of non-robustness** — primary statistic; large |Z| = fragile bias score
- **95% confidence intervals** using Student t-distribution: `μ_rep ± t_{R-1, 0.975} · σ_rep / √R`
- Significance codes: `ns`, `*`, `**`, `***`

---

## 📊 Key Results

- **All 4 models** show SWR systematically *lower* under noise than at baseline → bias scores on clean data are **optimistically inflated**
- **Z-scores exceed the p=0.05 threshold at Dose 1** for most model/dataset combinations → even light punctuation noise produces significant instability
- **Z-scores increase monotonically** with dose — fragility compounds with noise complexity
- **Mistral-7B-Instruct-v0.3** is the least robust model on CrowS-Pairs EN
- **Gemma-7B-IT** shows the largest SWR drop under max noise on StereoSet
- **English bias scores are less robust than French** across all models at Dose 2 — robustness is language- and dataset-dependent, not intrinsic to the model architecture
- **Gemma-7B-IT** shows near-zero, non-significant Z-score in French while remaining highly non-robust in English — strikingly different cross-lingual behavior

---

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/llm-bias-monte-carlo.git
cd llm-bias-monte-carlo

pip install transformers bitsandbytes accelerate datasets \
            nltk pandas numpy tqdm matplotlib seaborn huggingface_hub

python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
```

### HuggingFace Authentication
```bash
export HF_TOKEN=your_token_here
```

### Hardware
Experiments run on **NVIDIA A100 (80GB)** via Google Colab. 4-bit quantization via `BitsAndBytesConfig` allows all 7–8B models to fit in a single GPU context.

---

## 🚀 Pipeline

```
1. Data Loading            → CrowS-Pairs EN/FR (CSV/GitLab), StereoSet (HF Hub)
                             Cached to Google Drive via HF datasets
2. Dataset Standardization → Unified (context, anti, stereo) triplet format
3. Bias-Word Extraction    → Set difference between completions → protected tokens
4. Noise Augmentation      → 4 cumulative dose stacks (EN) + 1 stack (FR)
                             NLTK: tokenization, POS tagging, WordNet synonyms
5. Monte Carlo Simulation  → R replicas per (model, dataset, dose) triplet
                             Shared seed across models per replica
6. SWR Inference           → 4-bit quantized LLM log-prob extraction (A100)
                             Results saved per model to results_<model>.csv
7. Aggregation             → Combined CSV: final_combined_results_for_R.csv
8. Statistical Analysis    → Z-scores, p-values, CIs (Python + R/ggplot2)
9. Visualization           → Density plots, boxplots, dumbbell charts,
                             dose-response curves, multilingual bar charts
```

---

## 📁 Repository Structure

```
.
├── monte_carlo_llm_robustness.ipynb   # Main Colab pipeline (Python)
├── llm_monte_carlo_simulation.Rmd     # Statistical analysis & visualization (R/ggplot2)
├── README.md
├── report/
│   └── project_MC_2025_Yanis_Bencheikh_rapport.pdf
└── outputs/
    ├── final_combined_results_for_R.csv   # Aggregated simulation results
    └── figures/                           # Generated plots (density, boxplot, dumbbell, Z-score)
```

---

## 📖 References

1. Nangia et al. (2020). *CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models.* EMNLP 2020.
2. Nadeem et al. (2021). *StereoSet: Measuring Stereotypical Bias in Pretrained Language Models.* ACL-IJCNLP 2021.
3. Névéol et al. (2022). *French CrowS-Pairs: Extending a Challenge Dataset for Measuring Social Bias to French.* ACL 2022.
4. Gallegos et al. (2024). *Bias and Fairness in Large Language Models: A Survey.* Computational Linguistics.
5. Meade et al. (2021). *An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-Trained Language Models.* arXiv.
6. Robert & Casella (2004). *Monte Carlo Statistical Methods.* Springer.
7. Llama Team (2024). *The Llama 3 Herd of Models.* arXiv.
8. Gemma Team (2024). *Gemma: Open Models Based on Gemini Research and Technology.* arXiv.

---

## 👤 Author

| Author | Affiliation |
|---|---|
| **Yanis Bencheikh** | HEC Montréal, Mila – Quebec AI Institute |

Course: MATH 60603 — Automne 2025

---

## 📜 License

Released for academic and research purposes. See `LICENSE` for details.
