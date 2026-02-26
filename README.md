# Expedited Noise Spectroscopy of Transmon Qubits: ML Pipeline

This repository contains the machine learning pipeline developed to rapidly extract time-varying noise spectral densities associated with transmon qubits. 

Traditional noise spectroscopy protocols are often resource-intensive and struggle with time-dependent noise because they require deconvolving a measured coherence decay curve $C(t)$ with a known filter function $F(\omega t)$. This pipeline addresses that challenge by using a convolutional neural network (CNN) pre-trained on synthetic datasets to predict the underlying noise spectrum $S(\omega)$ nearly instantaneously with minimal absolute error.

## Repository Contents

* **`MAIN.py`**: The central training and evaluation script.
    * Loads `.npz` datasets.
    * Splits data into training/test sets using NumPy-only utilities.
    * Builds the CNN model variant specified by `--net_type`.
    * Trains the model using Mean Absolute Percentage Error (MAPE) loss and an Adam optimizer.
    * Saves trained weights and diagnostic plots to `TRAINED_NETWORKS/`.

* **`network_functions.py`**: Model architecture library.
    * Contains various 1D-CNN encoder-decoder architectures.
    * Features a dispatcher function `get_model(...)` to select architectures based on `net_type`.
    * All models utilize a `Dense(501)` output layer to represent the spectrum vector.

* **`data_generation_functions.py`**: Preprocessing and forward-modeling utilities.
    * `getFilter(...)`: Computes filter functions for dynamical decoupling sequences.
    * `getCoherence(...)`: A forward model that calculates coherence curves from spectra via numerical integration.
    * `prepare_trainData(...)`: Adds Gaussian noise to synthetic curves to emulate experimental signal-to-noise ratios.
    * `numpy_train_test_split(...)`: A utility for reproducible data splitting without external ML dependencies.

* **`bash_single_2.sh`**: A helper script to launch a single training job with optimized hyperparameters.

* **`bash_param_sweep.sh`**: A script for running automated hyperparameter sweeps across different filter counts and kernel sizes.

## Requirements

The pipeline requires the following Python libraries:
* `tensorflow` (Keras API)
* `numpy`
* `scipy` (for interpolation and numerical integration)
* `matplotlib` (for diagnostic plotting)
* `tqdm` (for progress tracking during data generation)

## Setup & Data Format

1. **Create Directory Structure**:
   ```bash
   mkdir data TRAINED_NETWORKS
   ```

2. **Data Preparation**:
   The `MAIN.py` script expects an `.npz` file in the `data/` folder. The file must contain the following keys:
   * `c_in`: Input coherence curves (N samples x Time points).
   * `s_in`: Target noise spectra (N samples x Frequency points).
   * `T_in` / `w_in`: Grids used for synthetic generation.
   * `T_train` / `w_train`: Grids used for model training/plotting.

> **Note**: The default dataset name is hardcoded as `Mar14_x32_noisy_20_noises` in `MAIN.py`. Update the `data_file_name` variable to match your filename.

## Running the Pipeline

### Training a Single Model
Execute `MAIN.py` with your desired hyperparameters:

```bash
python MAIN.py --batch_size 64 --epochs 20 --filters 40 --kernel_size 5 \
  --initial_lr 1e-3 --min_lr 1e-6 --patience 6 --min_delta 0.5 \
  --verbose 1 --net_type 1
```

### Performing a Parameter Sweep
To automatically test multiple architectures:
```bash
bash bash_param_sweep.sh
```

## Outputs

All results are saved in the `TRAINED_NETWORKS/` directory, labeled with a unique hyperparameter string:
* **`.h5` File**: The serialized Keras model weights and architecture.
* **`VAL_ACC_HISTORY_...pdf`**: Plot of training vs. validation loss (MAPE).
* **`MODEL_TEST_...pdf`**: Log-log plots of predicted vs. true noise spectra for random test samples.

## Reference
Gupta, B., Joshi, V., Kandpal, U., Mandayam, P., Gheeraert, N., & Dhomkar, S. (2025). **Expedited Noise Spectroscopy of Transmon Qubits**. *Advanced Quantum Technologies*. [DOI: 10.1002/qute.202500109](https://doi.org/10.1002/qute.202500109).
