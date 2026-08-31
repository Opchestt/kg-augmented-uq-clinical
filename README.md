# Knowledge Graph-Augmented Uncertainty Quantification for LLM-Based Clinical Diagnosis

> A knowledge graph-augmented uncertainty quantification (UQ) framework that combines LLM-derived confidence with structured biomedical evidence from PrimeKG for clinical diagnosis.

---

## Overview

Large Language Models (LLMs) are increasingly used to extract diagnoses from clinical notes, but quantifying the **reliability** of each predicted diagnosis remains a challenge. This project implements a knowledge graph-augmented uncertainty quantification (UQ) framework for LLM-based clinical diagnosis. The proposed method combines:

| Component | Description |
|-----------|-------------|
| **LLM Confidence** | Sequence-normalized mean token log probability obtained through teacher forcing |
| **KG-based Clinical Plausibility** | Maps clinical entities to PrimeKG and propagates evidence using Personalized Random Walk with Restart (RWR), with coherence- and specificity-aware anchor weighting |
| **KG Expansion** | Retrieves additional clinically plausible diseases from the KG and verifies them using the LLM before adding them to the candidate set |
| **Confidence Fusion** | Combines LLM confidence and KG-derived confidence using a weighted fusion coefficient $\gamma$ |

The framework uses PrimeKG as an external source of structured biomedical evidence rather than constructing a knowledge graph from generated responses.

### Evaluation

A reference confidence distribution is constructed using 100 stochastic generations per clinical note with Llama-3.1-8B-Instruct at temperature 0.6. Semantically equivalent disease entities are merged using bidirectional entailment with PubMedBERT-MNLI-MedNLI. The resulting frequency distribution serves as the reference ranking for evaluating uncertainty estimation.

The primary evaluation metrics are NDCG@5, RBO, and MRR. Additional metrics include NDCG@10, Precision/Recall@5/10, and Kendall's $\tau$.

---

## Pipeline Architecture

```
Clinical Note
      │
      ├── LLM Disease Generation
      │       └── Llama-3.1-8B-Instruct
      │               ↓
      │        Initial Disease Candidates
      │               │
      │               ├── LLM Confidence
      │               │      └── Mean Token Log Probability
      │               │
      │               └──────────────────────┐
      │                                      │
      └── Clinical Entity Extraction         │
              └── Qwen-2.5-72B-Instruct      │
                      ↓                      │
                SapBERT Mapping              │
                      ↓                      │
                  PrimeKG                    │
                      ↓                      │
             Coherence + Specificity         │
                      ↓                      │
                Personalized RWR             │
                      ↓                      │
               KG Score                      │
                      │                      │
                      └──── Confidence Fusion
                                   │
                                   ↓
                         KG Expansion
                                   │
                         LLM Verification
                                   │
                                   ↓
                         Final Ranking
```

---

## Repository Structure

```
.
├── step1_extract_entities.py   # Step 1: Extract clinical entities with Qwen-72B-AWQ
├── standard_sampling.py        # Baseline: Stochastic sampling UQ (100 samples)
├── step2_combined_uq.py        # Step 2: KG-Augmented UQ pipeline
├── uq_analysis.py              # Evaluation: Correlation & ranking metrics vs. sampling baseline
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md
├── data/                       # Input data (user-provided, not tracked)
│   ├── MIMIC_notes_icd_01.csv
│   └── primeKG/
└── results/                    # Output directory (auto-created)
    ├── mimic/
    ├── pmc/
    └── mtsamples/
```

### File Descriptions

| File | Purpose | Key Models Used |
|------|---------|-----------------|
| [`step1_extract_entities.py`](file:///home/hmilab/yuchieh/step1_extract_entities.py) |  Extracts clinically relevant entities from clinical narratives for KG-based evidence construction. | Qwen2.5-72B-Instruct |
| [`standard_sampling.py`](file:///home/hmilab/yuchieh/standard_sampling.py) | Generates 100 stochastic responses per note and constructs the reference confidence distribution by semantically clustering generated disease entities. | Llama-3.1-8B / Qwen-7B / Mistral-7B / Gemma-3-4B, PubMedBERT-MNLI-MedNLI |
| [`step2_combined_uq.py`](file:///home/hmilab/yuchieh/step2_combined_uq.py) | Core KG-Augmented UQ pipeline: computes LLM confidence, maps clinical entities to PrimeKG, performs coherence- and specificity-aware RWR, expands the candidate set, verifies expanded candidates, and fuses LLM and KG confidence. | Llama-3.1-8B / Qwen-7B / Mistral-7B / Gemma-3-4B, SapBERT, PubMedBERT-MNLI-MedNLI, PrimeKG |
| [`uq_analysis.py`](file:///home/hmilab/yuchieh/uq_analysis.py) | Evaluates UQ methods against the sampling-based reference using ranking and retrieval metrics. | Qwen2.5-72B-Instruct |

---

## Datasets

| Dataset | Description | Role |
|---------|-------------|------|
| **MIMIC-IV v3.1** | Clinical discharge summaries | Primary evaluation dataset |
| **PMC-Patients** | Clinical case reports | External evaluation dataset |
| **MTSamples** | Physician-dictated medical transcriptions | External evaluation dataset |

---

## External Resources

| Resource | Description | Usage |
|----------|-------------|-------|
| [PrimeKG](https://github.com/mims-harvard/PrimeKG) | Precision Medicine Knowledge Graph | Disease-phenotype-drug relationships for KG-augmented scoring |
| PrimeKG RotatE Embeddings | Pre-trained RotatE (dim=256) embeddings on PrimeKG | Coherence scoring for RWR personalization |
| [SapBERT](https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext) | Biomedical entity linking model | Mapping extracted entities to KG nodes |
| [PubMedBERT-MNLI-MedNLI](https://huggingface.co/pritamdeka/PubMedBERT-MNLI-MedNLI) | Medical NLI model | Bidirectional entailment for semantic equivalence checking |

---

## Setup

### 1. Data Directory

Place your input datasets and KG resources under a `data/` directory:

```
data/
├── MIMIC_notes_icd_01.csv          # MIMIC-IV discharge summaries
├── qwen_results_pmc.csv            # PMC-Patients (if applicable)
├── qwen_results_mtsamples.csv      # MTSamples (if applicable)
└── primeKG/
    ├── kg.csv                      # PrimeKG knowledge graph
    ├── primekg_rotate_dim256_embeddings.npy
    └── primekg_rotate_dim256_mapping.csv
```

### 2. Environment Variables

```bash
# Required: HuggingFace token for gated model access
export HF_TOKEN="your_huggingface_token"

# Optional: Override KG resource paths (defaults to data/primeKG/...)
export KG_PATH="/custom/path/to/kg.csv"
export KGE_EMB_PATH="/custom/path/to/embeddings.npy"
export KGE_MAP_PATH="/custom/path/to/mapping.csv"
```

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA)
- GPU: ≥ 24 GB VRAM recommended (A100/A6000); multi-GPU supported for 72B models

### Python Dependencies

```
torch
transformers
bitsandbytes
pandas
numpy
scipy
scikit-learn
networkx
tqdm
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Extract Entities

```bash
# MIMIC dataset, full mode
python step1_extract_entities.py --dataset mimic --mode full

# PMC dataset, sliced mode (cut off at Physical Exam)
python step1_extract_entities.py --dataset pmc --mode sliced

# Process specific range
python step1_extract_entities.py --dataset mimic --mode full --start_index 0 --end_index 100
```

### Step 2a: Standard Sampling (Baseline)

```bash
# Run with Llama-3.1-8B, 100 samples per note, batch size 10
python standard_sampling.py --dataset mimic --mode full --model llama --batch_size 10

# Run with alternative models
python standard_sampling.py --dataset mimic --mode full --model qwen
python standard_sampling.py --dataset mimic --mode full --model mistral
python standard_sampling.py --dataset mimic --mode full --model gemma
```

### Step 2b: Combined UQ Pipeline

```bash
# Run combined UQ (Logits + CCP + KG)
python step2_combined_uq.py --dataset mimic --mode full --model llama

# Customize KG parameters
python step2_combined_uq.py --dataset mimic --mode full --model llama \
    --alpha 0.25 \
    --top_k_ccp 5 \
    --top_k_kg 5 \
    --restart_rate 0.2 \
    --tf_threshold_type min_logprob \
    --tf_threshold_val -12.6
```

### Step 3: Evaluation

```bash
# Evaluate all methods against standard sampling baseline
python uq_analysis.py --dataset mimic --model llama

# Quick test with limited samples
python uq_analysis.py --dataset mimic --model llama --limit_index 50

# Ablation: ignore KG-expanded diseases
python uq_analysis.py --dataset mimic --model llama --ignore_expansion
```

---

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha` | 0.25 | Weight assigned to KG-derived confidence during LLM/KG fusion |
| `--top_k_kg` | 5 | Maximum number of KG-expanded disease candidates selected for LLM verification|
| `--restart_rate` | 0.2 | Restart probability in Personalized Random Walk with Restart |
| `--tf_threshold_type` | `min_logprob` | Threshold metric for filtering KG-expanded candidates |
| `--tf_threshold_val` | -12.6 | Threshold value for the selected metric |
| `--mode` | `full` | Input mode: `full` (complete note) or `sliced` (truncated at Physical Exam) |

---

## Output Format

### Standard Sampling (`standard_sampling_*.csv`)

| Column | Description |
|--------|-------------|
| `index` | Note index |
| `disease` | Cluster representative disease name |
| `uq` | Frequency-based UQ score (0–1) |

### Combined UQ (`combined_uq_*.csv`)

| Column | Description |
|--------|-------------|
| `index` | Note index |
| `disease` | Extracted disease name |
| `uq_pure` | Geometric mean of token probabilities |
| `uq_seq_prob` | Product of token probabilities |
| `uq_neg_ent` | Negative mean entropy |
| `uq_max_logprob` | Maximum token log-probability |
| `uq_min_logprob` | Minimum token log-probability |
| `uq_ccp` | CCP baseline confidence score |
| `uq_kg` | Final KG-augmented confidence score |
| `kg_only` | Pure KG score (normalized RWR probability) |
| `is_expanded` | Whether this disease was discovered via KG expansion |

### Evaluation (`correlation_summary_results_*.csv`)

| Column | Description |
|--------|-------------|
| `Method` | UQ method name |
| `NDCG@5/10/std` | Normalized Discounted Cumulative Gain |
| `Kendall_tau_b` | Kendall's rank correlation |
| `MRR` | Mean Reciprocal Rank |
| `RBO` | Rank-Biased Overlap (p=0.9) |
| `Precision/Recall@5/10` | Precision and Recall at K |

---

## Note Preprocessing

Both `full` and `sliced` modes apply section removal to prevent information leakage:

- **Removed sections**: Past Medical History, Social/Family History, Discharge Diagnosis/Medications, Medications on Admission
- **Sliced mode** additionally truncates the note at the Physical Examination section, simulating a real-time clinical scenario where only pre-exam information is available

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{anonymous2026kg_uq,
  title     = {Knowledge Graph-Augmented Uncertainty Quantification for LLM-Based Clinical Diagnosis},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of Machine Learning for Health},
  year      = {2026}

}
```

> **Note:** Citation details will be updated upon publication.

---

## License

This project is licensed under the [MIT License](LICENSE).
