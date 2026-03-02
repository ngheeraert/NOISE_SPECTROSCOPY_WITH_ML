#!/bin/bash

# =============================================================================
# Script: Single Training Run Launcher
#
# Purpose:
# Executes a single instance of MAIN.py with static hyperparameters. 
#
# Physical Context:
# This script specifically targets network construction and training. 
# =============================================================================

# --- CONFIGURATION ---
# Defines the training constraints and network dimensions.

BATCH_SIZE=64        # Number of samples per gradient update
EPOCHS=2             # Limited pass count for quick functional verification
INITIAL_LR=0.001     # Starting learning rate for the Adam optimizer
MIN_LR=0.000001      # Lower bound for the learning rate schedule
PATIENCE=6           # Epochs to wait for improvement before LR reduction
MIN_DELTA=0.1        # Minimum change in loss to qualify as an improvement
VERBOSE=1            # Enables detailed logging of the training progress
NET_TYPE=3           # Selects the 4-stage deeper Encoder-Decoder variant

FILTERS=80           # High filter count for capturing dense spectral features
KERNEL_SIZE=21       # Wide kernel to capture long-range temporal correlations in C(t)

# --- EXECUTION ---
# Launches the Python training entry point with the above arguments.

echo " "
echo "-- Starting Noise Spectroscopy Training..."
echo "-- Config: NetType=$NET_TYPE, Filters=$FILTERS, Kernel=$KERNEL_SIZE"

python MAIN.py --batch_size $BATCH_SIZE --epochs $EPOCHS --filters $FILTERS \
	--kernel_size $KERNEL_SIZE --initial_lr $INITIAL_LR --min_lr $MIN_LR \
	--patience $PATIENCE --min_delta $MIN_DELTA --verbose $VERBOSE --net_type $NET_TYPE


