# Day 5 — MLflow Experiment Tracking + Federated Learning

**AI Skills Bootcamp | Tsegai Yhdego, PhD**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-blue)](https://mlflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-orange)](https://pytorch.org)
[![FedAvg](https://img.shields.io/badge/Algorithm-FedAvg-green)](https://github.com/Tsegai20)

---

## What I Built

Two experiments combining MLflow experiment tracking with federated learning simulation — directly mirroring the Non-IID federated learning challenges studied in my PhD dissertation on attribute-based cognitive ML for manufacturing quality control.

---

## Experiment 1 — Centralized BERT Baseline

Standard fine-tuning on 1000 balanced samples from all 4 classes.

| Metric | Value |
|--------|-------|
| Epoch 1 accuracy | 77.7% |
| Epoch 2 accuracy | **84.3%** |
| Final eval loss | 0.508 |
| Training time | ~47 minutes CPU |

**Key finding:** Centralized training with balanced data achieves strong performance across all four classes. This is the upper bound target for federated learning to approach.

---

## Experiment 2 — Federated Learning (Non-IID, 3 Sites, 1 Round)

### Data Distribution

| Site | Classes | Samples | Description |
|------|---------|---------|-------------|
| Site A | World only | 800 | Extreme single-class skew |
| Site B | Sports + Business | 800 (400 each) | Two-class distribution |
| Site C | Sci/Tech only | 800 | Extreme single-class skew |
| Global Test | All 4 classes | 600 (150 each) | Balanced evaluation |

No site shares raw data. Only model weights are shared via FedAvg.

### Results

| Model | Overall Accuracy | Key Observation |
|-------|-----------------|-----------------|
| Site A (World only) | 25.0% | Predicts everything as World |
| Site B (Sports+Business) | 49.3% | Best local — two classes |
| Site C (Sci/Tech only) | 25.0% | Predicts everything as Sci/Tech |
| **Global FedAvg** | **33.3%** | Dominated by Sci/Tech class |

### Global Model Classification Report

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| World | 0.22 | 0.01 | 0.03 |
| Sports | 0.98 | 0.28 | 0.44 |
| Business | 0.75 | 0.06 | 0.11 |
| Sci/Tech | 0.27 | 0.98 | 0.43 |
| **Overall** | **0.56** | **0.33** | **0.25** |

---

## Experiment 3 — Multi-Round FedAvg (5 Rounds, 4 Sites)

Running multiple rounds dramatically reduces client drift.

| Round | Global Accuracy | Improvement |
|-------|----------------|-------------|
| 1 | 39.8% | +39.8% |
| 2 | 50.5% | +10.7% |
| 3 | 58.2% | +7.7% |
| 4 | 63.7% | +5.5% |
| 5 | **66.7%** | +3.0% |

**Convergence pattern:** Rapid initial improvement then diminishing returns — exactly what FL theory predicts.

**Non-IID gap vs centralized:** 84.3% - 66.7% = **17.6 percentage points**

---

## Key Findings

### Finding 1 — Non-IID data causes client drift
Each local model converges toward its own data distribution. Site A predicts everything as World. Site C predicts everything as Sci/Tech. This is the fundamental challenge of federated learning under extreme Non-IID conditions.

### Finding 2 — Single-round FedAvg fails under extreme Non-IID
The global model was dominated by Sci/Tech predictions because Site C had the strongest single-class signal. FedAvg equal weighting did not account for class imbalance across sites.

### Finding 3 — Multi-round FedAvg significantly reduces client drift
Five rounds improved accuracy from 33.3% to 66.7% — a 33 percentage point improvement. The improvement follows a classic convergence curve with diminishing returns per round.

### Finding 4 — This directly mirrors my PhD research
Traditional FedAvg breaks down under extreme Non-IID conditions, which are common in manufacturing where different sites produce different product types. My dissertation addresses this with attribute-based cognitive ML enabling knowledge transfer at the semantic attribute level rather than raw weight averaging.

---

## MLflow Experiment Tracking

All experiments tracked with full reproducibility:

```python
with mlflow.start_run(run_name="federated_3sites_noniid"):
    mlflow.log_params({
        "num_sites": 3,
        "data_distribution": "Non-IID",
        "algorithm": "FedAvg",
        "model": "bert-base-uncased"
    })
    mlflow.log_metrics({
        "global_accuracy": global_accuracy,
        "improvement": improvement
    })
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("classification_report.txt")
```

**Tracked per experiment:**
- Parameters: model, data distribution, site assignments, sample counts
- Metrics: per-site accuracy, global accuracy, per-class accuracy per round
- Artifacts: confusion matrices, classification reports, convergence plots
- Tags: experiment type, algorithm, developer, research relevance

---

## FedAvg Implementation

```python
def fedavg(model_states, weights=None):
    if weights is None:
        weights = [1.0 / len(model_states)] * len(model_states)
    
    averaged = {}
    for key in model_states[0].keys():
        averaged[key] = sum(
            w * state[key].float()
            for w, state in zip(weights, model_states)
        )
    return averaged
```

---

## Future Work

| Improvement | Expected Impact |
|-------------|----------------|
| FedProx — proximal term in local objective | Reduces client drift |
| Class-balanced FedAvg weighting | Reduces Sci/Tech dominance |
| Differential privacy noise | Privacy-preserving FL |
| More federated rounds | Continued convergence |
| Attribute-based aggregation | PhD dissertation approach |

---

## Project Structure

day5/
├── federated_learning_report.md # This document
├── experiment_summary.json # All run IDs and metrics
├── federated_convergence.png # 5-round convergence chart
├── mlflow.db # MLflow SQLite tracking
└── README.md


---

## Technical Stack

- **Model:** BERT base uncased (bert-base-uncased)
- **Framework:** HuggingFace Transformers · PyTorch
- **Aggregation:** FedAvg implemented from scratch
- **Tracking:** MLflow with SQLite backend
- **Evaluation:** Scikit-learn classification report
- **Visualization:** Matplotlib · Seaborn

---

## Skills Added to Resume

- MLflow experiment tracking — parameters, metrics, artifacts, tags
- FedAvg implementation from scratch
- Non-IID federated learning simulation
- Client drift diagnosis and analysis
- Multi-round convergence measurement
- Confusion matrix analysis for model bias detection
- Reproducible ML experiment design

---

## Part of AI Skills Bootcamp

| Day | Project | Repo |
|-----|---------|------|
| 1 | Streamlit AI Apps | [Day 1](https://github.com/Tsegai20/ai-skills-bootcamp) |
| 2 | Fine-Tuned BERT Classifier | [Day 2](https://github.com/Tsegai20/ai-skills-bootcamp) |
| 3 | FastAPI ML Serving | [Day 3](https://github.com/Tsegai20/ai-skills-bootcamp) |
| 4 | RAG Pipeline | [Day 4](https://github.com/Tsegai20/ai-skills-bootcamp) |
| **5** | **MLflow + Federated Learning** | **You are here** |
| 6 | Docker Federated System | [ai-skills-bootcamp6](https://github.com/Tsegai20/ai-skills-bootcamp6) |

---

## Research Connection

This experiment directly mirrors challenges studied in my PhD dissertation on **attribute-based cognitive ML for manufacturing quality control**:

- Non-IID data → different manufacturing sites produce different product types
- Privacy preservation → raw production data never leaves each site
- Client drift → local models bias toward local data distribution
- Multi-round convergence → iterative aggregation reduces drift
- 17.6% gap → motivates advanced FL algorithms including attribute-based learning

**Published:** IEEE T-ASE 2025 · ASME JCISE 2024  
**Patent filed:** Privacy-preserving federated learning system

---

## Author

**Tsegai Yhdego, PhD**  
Industrial Engineering — Machine Learning and Optimization  
Assistant Director, R-SEAT Center, FAMU-FSU  
USDOT-Funded Research — Intelligent Transportation and Autonomous Systems  
[GitHub](https://github.com/Tsegai20) · [LinkedIn](https://www.linkedin.com/in/tsegai-yhdego-ph-d-b6209b118/)
