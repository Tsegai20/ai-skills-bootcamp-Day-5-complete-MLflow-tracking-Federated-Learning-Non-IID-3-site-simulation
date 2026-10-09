#MLflow
# Federated Learning Experiment Report
## Non-IID Multi-Site BERT Text Classification

**Author:** Tsegai Yhdego, PhD  
**Institution:** R-SEAT Center, FAMU-FSU  
**Date:** October 2026  
**Framework:** PyTorch, HuggingFace Transformers, MLflow  

---

## Experiment Overview

This experiment simulates a federated learning setup across three geographically 
distributed sites with Non-IID (non-independent and identically distributed) data. 
The setup directly mirrors real-world federated learning challenges studied in 
Tsegai's PhD dissertation on attribute-based cognitive ML for manufacturing quality 
control.

---

## Dataset

**AG News** — 120,000 news articles across 4 categories:
- 0: World news
- 1: Sports news  
- 2: Business news
- 3: Sci/Tech news

---

## Non-IID Data Distribution

| Site | Classes | Samples | Description |
|------|---------|---------|-------------|
| Site A | World only | 800 | Extreme single-class skew |
| Site B | Sports + Business | 800 (400 each) | Two-class distribution |
| Site C | Sci/Tech only | 800 | Extreme single-class skew |
| Global Test | All 4 classes | 600 (150 each) | Balanced evaluation set |

No site shares raw data with any other site. Only model weights are shared.

---

## Experiments Conducted

### Experiment 1 — Baseline BERT (Centralized)
Standard fine-tuning on 1000 mixed samples from all classes.

**Results:**
- Epoch 1 accuracy: 77.7%
- Epoch 2 accuracy: 84.3%
- Final eval loss: 0.508
- Training time: ~47 minutes on CPU

**Key finding:** Centralized training with balanced data achieves strong performance 
across all four classes.

---

### Experiment 2 — Federated Learning (Non-IID, 3 Sites, 1 Round)
FedAvg aggregation after one round of local training at each site.

**Local Model Results:**

| Model | Overall Accuracy | Key Observation |
|-------|-----------------|-----------------|
| Site A (World only) | 25.0% | Predicts everything as World |
| Site B (Sports+Business) | 49.3% | Best local model — two classes |
| Site C (Sci/Tech only) | 25.0% | Predicts everything as Sci/Tech |
| Global Federated Model | 33.3% | Dominated by Sci/Tech class |

**Global Model Classification Report:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| World | 0.22 | 0.01 | 0.03 |
| Sports | 0.98 | 0.28 | 0.44 |
| Business | 0.75 | 0.06 | 0.11 |
| Sci/Tech | 0.27 | 0.98 | 0.43 |
| **Overall** | **0.56** | **0.33** | **0.25** |

---

## Key Findings and Analysis

### Finding 1 — Non-IID data causes client drift
Each local model converged toward its own data distribution:
- Site A predicted almost all samples as World
- Site C predicted almost all samples as Sci/Tech
- Site B performed best locally because it had two classes

### Finding 2 — Single round FedAvg fails under extreme Non-IID
The global model was dominated by Sci/Tech predictions because:
- Site C had the strongest single-class signal (800 pure Sci/Tech samples)
- FedAvg equal weighting did not account for class imbalance across sites
- One round of local training caused severe client drift before aggregation

### Finding 3 — Sci/Tech dominated the federated average
Confusion matrix shows 144/150 World, 108/150 Sports, 137/150 Business,
and 147/150 Sci/Tech articles all predicted as Sci/Tech after aggregation.

---

## Why This Matters

This experiment demonstrates the core challenge that Tsegai's PhD dissertation 
addressed — traditional FedAvg breaks down under extreme Non-IID conditions, 
which are common in real manufacturing settings where:
- Different sites produce different product types
- Local data distributions are highly skewed
- Class coverage varies dramatically across sites

The solution proposed in Tsegai's research — attribute-based cognitive ML with 
transferable attribute learning — addresses this by enabling knowledge transfer 
across sites at the semantic attribute level rather than raw weight averaging.

---

## Known Limitations and Future Work

1. **Single aggregation round** — Multiple rounds with smaller local epochs 
   (FedProx or multi-round FedAvg) would reduce client drift significantly

2. **Equal weighting** — Class-balanced weighting in FedAvg would reduce 
   Sci/Tech dominance

3. **No differential privacy** — Adding DP noise to weight updates would 
   demonstrate privacy-preserving federated learning

4. **FedProx** — Adding proximal term to local training objective prevents 
   local models from drifting too far from global model

---

## MLflow Experiment Tracking

All experiments tracked in MLflow with:
- Parameters: model, data distribution, site assignments, sample counts
- Metrics: per-site accuracy, global accuracy, per-class accuracy
- Artifacts: confusion matrices, classification reports
- Tags: experiment type, algorithm, developer, research relevance

**Experiment 1 Run ID:** 4e59bb12bfde4998bf9828cc15abccc2  
**Experiment 2 Run ID:** e70042d891a7463698dfc1fb298c6bf8  

---

## Technical Stack

- **Model:** BERT base uncased (bert-base-uncased)
- **Framework:** HuggingFace Transformers, PyTorch
- **Aggregation:** FedAvg (Federated Averaging)
- **Tracking:** MLflow with SQLite backend
- **Evaluation:** Scikit-learn classification report
- **Visualization:** Matplotlib, Seaborn

---

## Conclusion

This experiment successfully demonstrates both the power and limitations of 
federated learning under Non-IID conditions. The centralized baseline achieved 
84.3% accuracy with balanced data, while the federated approach with extreme 
Non-IID distribution and single-round FedAvg achieved only 33.3% — highlighting 
the client drift problem that motivates more advanced federated learning algorithms 
and attribute-based approaches developed in Tsegai's PhD research.
