# GNN for Money Laundering Detection

Final project for UCLA STATS 426 (Deep Learning), comparing graph-based
approaches for detecting illicit transactions in a large, heavily imbalanced
financial transaction network.

## Dataset

[IBM AML (Anti-Money Laundering) transaction dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml),
specifically the `HI-Small_Trans.csv` split — synthetic bank transactions
where each row is a transfer between two accounts, labeled as laundering
("is laundering") or not. The dataset is not included in this repo; download
it from Kaggle and update the file paths in the notebooks/scripts to point
to your local copy.

The dataset is heavily imbalanced (laundering transactions are a small
minority), which shapes most of the modeling decisions below.

## Approach

- **`notebooks/STATS426_Project_Initial_EDA.ipynb`** — exploratory analysis
  of the raw dataset: schema checks, null checks, and class imbalance.
- **`src/xgboost_baseline.py`** — a gradient-boosted tree baseline. Builds a
  transaction graph with `networkx` and engineers graph features (PageRank,
  HITS hub/authority scores, in/out-degree centrality) per account, then
  trains an XGBoost classifier on top of them.
- **`notebooks/Deep_Learning_Project_FraudGT_Final.ipynb`** — a Graph
  Transformer (FraudGT) trained directly on the transaction graph.
- **`notebooks/Deep_Learning_Project_MultiPNA_Final.ipynb`** — a GNN using
  Principal Neighbourhood Aggregation (PNA), with undersampling to address
  both class imbalance and GPU memory constraints.

## Results

| Model | F1 | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| FraudGT (Graph Transformer) | 0.315 | – | – | 0.962 | – |
| Multi-PNA (GNN) | 0.557 | 0.731 | 0.449 | 0.978 | 0.481 |

Multi-PNA was the strongest model overall, trading some recall for much
higher precision than FraudGT at a tuned decision threshold. Full training
curves, ablations, and discussion are in `docs/STATS_426_Final_Report.pdf`
and `docs/STATS_426_Final_Presentation.pptx`.

## Repo structure

```
notebooks/   EDA + model notebooks (FraudGT, Multi-PNA)
src/         XGBoost baseline script
docs/        Final report (PDF) and presentation (PPTX)
```

## Setup

The notebooks were developed in Google Colab and expect PyTorch, PyTorch
Geometric, `networkx`, `xgboost`, and standard data-science libraries
(`pandas`, `numpy`, `scikit-learn`). Update the hardcoded dataset paths at
the top of each notebook/script to point to your local copy of
`HI-Small_Trans.csv` before running.
