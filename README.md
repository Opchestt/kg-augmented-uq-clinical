# Knowledge Graph-Augmented Uncertainty Quantification for LLM-Based Clinical Diagnosis

> Uncertainty Quantification (UQ) framework combining **token-level logits analysis**, **Conformal Candidate Prediction (CCP)**, and **Knowledge Graph (KG) random walk** to assess the reliability of LLM-generated disease diagnoses from clinical notes.

---

## Overview

Large Language Models (LLMs) are increasingly used to extract diagnoses from clinical notes, but quantifying the **reliability** of each predicted diagnosis remains a challenge. This project proposes a unified UQ pipeline that fuses three complementary signals:

| Signal | Description |
|--------|-------------|
| **Pure Logits** | Token-level probability metrics (geometric mean, sequence probability, entropy, min/max log-prob) computed via teacher forcing |
| **CCP (Conformal Candidate Prediction)** | Measures semantic stability by branching at the first token of each diagnosis and checking if top-k alternatives remain semantically equivalent (via NLI) |
| **KG-Augmented** | Maps diagnoses onto PrimeKG, runs Personalized PageRank / Random Walk with Restart (RWR), and scores each diagnosis by its topological plausibility |

A **Reciprocal Rank Fusion (RRF)** score additionally merges Pure Logits and KG rankings.

### Evaluation

The ground truth is constructed via **Standard Sampling** (100 stochastic samples per note) to approximate the true marginal probability of each diagnosis. All UQ methods are then evaluated against this baseline using NDCG, Kendall-τ, MRR, RBO, Precision/Recall@K, with a Qwen-72B judge for cross-method entity alignment.

---

## Pipeline Architecture

```
Clinical Note
      │
      ├──── Step 1: Entity Extraction (Qwen-72B-AWQ)
      │         └── Extract diseases, symptoms, PMH from note → JSONL
      │
      ├──── Standard Sampling (Baseline)
      │         └── 100× stochastic generation (Llama-3.1-8B / Qwen-7B / Mistral-7B / Gemma-3-4B)
      │             → NLI-based clustering → frequency-based UQ score
      │
      └──── Step 2: Combined UQ Pipeline
                ├── [A] Pure Logits (Teacher Forcing)
                ├── [B] CCP (Top-K branching + NLI)
                ├── [C] KG-Augmented (PrimeKG + SapBERT mapping + RWR)
                ├── [D] RRF Fusion
                └── [E] KG Expansion (discover unseen diagnoses via RWR)
                        └── Filtered by Teacher Forcing threshold
```

---

## Repository Structure

```
.
├── step1_extract_entities.py   # Step 1: Extract clinical entities with Qwen-72B-AWQ
├── standard_sampling.py        # Baseline: Stochastic sampling UQ (100 samples)
├── step2_combined_uq.py        # Step 2: Unified UQ pipeline (Logits + CCP + KG + RRF)
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
| [`step1_extract_entities.py`](file:///home/hmilab/yuchieh/step1_extract_entities.py) | Uses Qwen-72B-AWQ to extract structured clinical entities (diseases, symptoms, PMH) from discharge summaries. Outputs a JSONL file used as KG context seeds in Step 2. | Qwen2.5-72B-Instruct-AWQ |
| [`standard_sampling.py`](file:///home/hmilab/yuchieh/standard_sampling.py) | Generates 100 stochastic samples per note, parses disease lists, clusters semantically equivalent diseases via bidirectional NLI, and computes frequency-based UQ scores as ground truth. | Llama-3.1-8B / Qwen-7B / Mistral-7B / Gemma-3-4B, PubMedBERT-MNLI-MedNLI |
| [`step2_combined_uq.py`](file:///home/hmilab/yuchieh/step2_combined_uq.py) | Core UQ pipeline: (A) Teacher-forced logits metrics, (B) CCP via top-k branching + NLI, (C) KG-augmented scoring via SapBERT entity linking + PrimeKG RWR, (D) RRF fusion, and (E) KG-based disease expansion with TF threshold filtering. | Llama-3.1-8B / Qwen-7B / Mistral-7B / Gemma-3-4B, SapBERT, PubMedBERT-MNLI-MedNLI, PrimeKG |
| [`uq_analysis.py`](file:///home/hmilab/yuchieh/uq_analysis.py) | Evaluates all UQ methods against the standard sampling baseline. Uses Qwen-72B as an alignment judge for cross-method entity matching. Computes NDCG@K, Kendall-τ, MRR, RBO, Precision/Recall@K. | Qwen2.5-72B-Instruct-AWQ |

---

## Datasets

| Dataset | Source | Text Column |
|---------|--------|-------------|
| **MIMIC-III** | Discharge summaries from MIMIC-III | `discharge_summary` |
| **PMC-Patients** | Structured case reports from PubMed Central | `structured_output` |
| **MTSamples** | Medical transcription samples | `structured_output` |

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
├── MIMIC_notes_icd_01.csv          # MIMIC-III discharge summaries
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
# Run combined UQ (Logits + CCP + KG + RRF)
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
| `--alpha` | 0.25 | Blending weight for KG score in KG-augmented UQ: `α·KG + (1-α)·Logits` |
| `--top_k_ccp` | 5 | Number of alternative first-token candidates for CCP |
| `--top_k_kg` | 5 | Maximum number of KG-expanded disease candidates |
| `--restart_rate` | 0.2 | Restart probability for Random Walk with Restart |
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
| `uq_ccp` | Conformal Candidate Prediction score |
| `uq_kg` | KG-augmented UQ score (α·KG + (1-α)·Logits) |
| `kg_only` | Pure KG score (normalized RWR probability) |
| `uq_rrf` | Reciprocal Rank Fusion of Pure Logits + KG |
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
