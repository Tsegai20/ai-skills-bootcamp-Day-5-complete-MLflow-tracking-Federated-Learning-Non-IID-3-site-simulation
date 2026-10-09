# Day 5 — MLflow Experiment Tracking and Federated Learning

## What I built
Complete ML experiment tracking system using MLflow combined with a 
federated learning simulation across 3 Non-IID sites.

## Experiments

### Experiment 1 — Centralized BERT Baseline
- Dataset: AG News, 1000 balanced samples
- Model: BERT base uncased fine-tuned
- Result: 84.3% accuracy across all 4 classes
- Tracked: parameters, metrics, confusion matrix, classification report

### Experiment 2 — Federated Learning Non-IID 3 Sites
- Site A: World news only (800 samples)
- Site B: Sports + Business (400 each)
- Site C: Sci/Tech only (800 samples)
- Algorithm: FedAvg (1 round)
- Global model accuracy: 33.3%
- Key finding: Client drift under extreme Non-IID conditions

## Key Findings
- Centralized balanced training: 84.3%
- Non-IID federated (1 round): 33.3%
- Client drift caused Sci/Tech dominance in aggregated model
- This motivates advanced FL algorithms and attribute-based approaches

## What I learned
- MLflow experiment tracking — parameters, metrics, artifacts, tags
- FedAvg implementation from scratch
- Non-IID data distribution effects on federated learning
- Client drift problem in federated settings
- Confusion matrix analysis for diagnosing model bias

## Tech Stack
- MLflow with SQLite backend
- HuggingFace Transformers
- PyTorch
- FedAvg from scratch
- Scikit-learn evaluation metrics
- Matplotlib and Seaborn visualization

## Author
Tsegai Yhdego
PhD Industrial Engineering — FAMU-FSU
AI/ML Researcher — R-SEAT Center
USDOT Funded Research
