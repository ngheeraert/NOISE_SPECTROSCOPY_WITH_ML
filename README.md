# Noise Spectroscopy ML Pipeline (commented)

This folder contains a small training pipeline that:
- loads a dataset of **coherence curves** (time-series inputs) and corresponding **noise spectra** (vector targets),
- trains a **1D convolutional neural network** to map `coherence(t) -> spectrum(ω)`,
- saves the trained model and a couple of diagnostic plots.

## Repository contents

- `MAIN.py`  
  End-to-end training/evaluation script:
  - loads `data/<dataset>.npz`,
  - splits into train/test,
  - builds a model variant selected by `--net_type`,
  - trains with a learning-rate scheduler callback,
  - saves the model and plots under `TRAINED_NETWORKS/`.

- `network_functions.py`  
  Model zoo / architecture definitions:
  - `get_model(...)` dispatches to one of several 1D-CNN variants via `net_type`.
  - Output head is a fixed-length spectrum vector (`Dense(501)` in the provided architectures).

- `data_generation_functions.py`  
  Utilities to preprocess / synthesize data:
  - NumPy-only train/test split (`numpy_train_test_split`),
  - interpolation helpers to map data between grids,
  - simple forward-model utilities (filter functions / coherence evaluation),
  - `generate_final_data(...)` to augment coherence curves with small additive noise.

- `bash_single_2.sh`  
  Runs a single training job with fixed hyperparameters.

- `bash_param_sweep.sh`  
  Runs repeated trainings while randomly sampling architecture parameters (filters, kernel size).

## Requirements

The scripts import the following common Python packages:
- `tensorflow` (Keras API)
- `numpy`
- `matplotlib`
- `scipy` (for interpolation in `data_generation_functions.py`)
- `tqdm` (progress bars in `data_generation_functions.py`)

## Data format

`MAIN.py` expects an `.npz` file in a `data/` subfolder with (at minimum) these arrays:

- `c_in`: input coherence curves, shape `(N, Nt)` (or `(N, Nt, 1)` depending on how you saved it)
- `s_in`: target spectra, shape `(N, Nw)`
- `T_in`: time grid used to generate `c_in`
- `w_in`: frequency grid used to generate `s_in`
- `T_train`: time grid used for model input / plotting
- `w_train`: frequency grid used for model output / plotting

By default, `MAIN.py` hard-codes a dataset name (see `data_file_name=...` in the script). To use a different dataset,
update that string so it matches your `.npz` filename in `data/`.

## Running a single training

From the folder containing `MAIN.py`:

```bash
python MAIN.py --batch_size 64 --epochs 20 --filters 40 --kernel_size 5 \
  --initial_lr 1e-3 --min_lr 1e-6 --patience 6 --min_delta 0.5 \
  --verbose 1 --net_type 3
```

Or run the provided launcher:

```bash
bash bash_single_2.sh
```

## Parameter sweeps

The sweep script repeatedly calls `MAIN.py` while randomly choosing `FILTERS` and `KERNEL_SIZE` within the ranges
defined in `bash_param_sweep.sh`:

```bash
bash bash_param_sweep.sh
```

Stop the sweep with `Ctrl+C`.

## Outputs

Training outputs are written to `TRAINED_NETWORKS/` (created if it does not exist):
- `MODEL_<paramchar>.h5` : saved Keras model
- `VAL_ACC_HISTORY_<paramchar>.pdf` : training/validation loss history plot
- `MODEL_TEST_<paramchar>.pdf` : a small random subset of predicted vs. true spectra (log-log)

The `<paramchar>` string is constructed from hyperparameters and the final validation loss to make runs easy to identify.

## Reference

B. Gupta et al., “Expedited Noise Spectroscopy of Transmon Qubits”, *Advanced Quantum Technologies* (2025), DOI: 10.1002/qute.202500109
