# Neural network experiments

The neural network code uses the functions in `src.data_loader` to load the
data, split it into training and validation sets, and standardize it. The split
always uses seed 42. I picked the settings in every version using the
validation results only. The test data is only used to make the final
predictions.

To install the packages:

```powershell
python -m pip install -r requirements.txt
```

To run versions 1 to 4 for both tasks (it uses the GPU if there is one):

```powershell
python run_experiments.py
```

Some other ways to run it:

```powershell
python run_experiments.py --task image --version 3 --epochs 20 --device cpu
python run_experiments.py --task sentiment --version 4 --device cuda
```

`--version N` runs versions 1 to N in order, because each version starts from
the settings the previous one picked. `--epochs` is the most epochs each
validation run can train for (V3 can stop earlier because of early stopping).
For the final predictions, the model is trained again on the training and
validation data together, for the number of epochs that did best on
validation.

What each version changes:

- V1: the baseline network.
- V2: the learning rate and the batch size, one at a time.
- V3: dropout and weight decay, one at a time.
- V4: the number of neurons per hidden layer and the number of hidden layers,
  one at a time.

The results are saved the same way as the k-NN results. Each task has
`results/<task>/nn_vN_summary.csv` and `nn_vN_validation.csv`. The combined
summary and the accuracy plots are in `results/nn/`. When both tasks are run,
it also makes `submission/nn/submission_vN_nn.csv` in the same ID order as the
sample submission.

To make the confusion and F1 tables for the report after running the
experiments:

```powershell
python -m src.analysis_nn
```

These tables are saved in `results/nn/`. If my teammates'
validation files are in `predictions/`, it also compares which examples each
model got wrong.

### Running a version from its notebook

Each version from `notebooks/neural_network_v1.ipynb` to
`notebooks/neural_network_v4.ipynb` can be run on its own in Jupyter or VS Code.
You can set `TASK`, `EPOCHS` and `DEVICE` at the top. Versions 2 to 4 run the
earlier versions first so they start from the right settings.

Versions 6 to 9 are separate notebooks
(`notebooks/neural_network_v6_epoch_experiments.ipynb` to
`notebooks/neural_network_v9_regularization_experiments.ipynb`) and each one
runs by itself.
