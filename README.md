# COMP 472 Project

## Setup

Put the Kaggle CSV files in `data/`, then install the packages:

```powershell
python -m pip install -r requirements.txt
```

## How to run

Open a notebook in `notebooks/` and run all cells. Every notebook runs on its
own except neural network V1 to V4 (see below).

| Model | Notebooks | Code |
|---|---|---|
| k-NN | `knn_experiments.ipynb` (V1), `knn_v2` to `knn_v9`, `knn_final.ipynb` | `src/models/knn.py` |
| Decision tree | `decision_tree_v1` to `decision_tree_v5` | `src/models/decision_tree.py` |
| Random forest | `random_forest_v1` to `random_forest_v5`, `random_forest_final.ipynb` | `src/models/random_forest.py` |
| Clustering | `clustering_v1` to `clustering_v3` | `src/models/clustering.py` |
| Neural network | `neural_network_v1` to `v4`, `v6` to `v9` | `src/neural_net.py` |

All models load and split the data with `src/data_loader.py` (seed 42).

## Neural network versions



V1 to V4 build on each other, so each one runs the earlier versions first. You
can also run them from the terminal with `python run_experiments.py`.

## Output

- `results/<model>/` and `results/<task>/`: summaries and validation predictions
- `submission/<model>/`: Kaggle submission files
