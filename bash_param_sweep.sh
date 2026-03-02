# =============================================================================
# Script: Parameter Sweep Launcher for Automated Architecture Search
#
# Purpose:
# Automates the exploration of the neural network's hyperparameter space. 
# This script repeatedly executes MAIN.py while randomly sampling key structural 
# parameters to identify the configuration that minimizes prediction error for noise spectra.
#
#
# Physical & Theoretical Context:
# Over-parametrized neural networks tend to obtain global minima with high.
# This sweep implements a brute-force search to find 
# the optimal filter-kernel combination that best characterizes the 1/f and 
# white noise regimes inherent in transmon qubits.
# =============================================================================

# --- OPTIMIZATION CONSTRAINTS ---
# Fixed training parameters used for all runs in the sweep.

BATCH_SIZE=64        # Number of samples per gradient update
EPOCHS=20            # Full training duration for convergence
INITIAL_LR=0.001     # Starting learning rate for the Adam optimizer
MIN_LR=0.00005       # Floor for the learning rate reduction schedule
PATIENCE=4           # Epochs to wait for improvement before reducing LR
MIN_DELTA=0.5        # Minimum change to qualify as a loss reduction
VERBOSE=False        # Suppresses detailed logging for cleaner sweep output

# --- SAMPLING BOUNDARIES ---
# Defines the range for random architectural exploration.

max_fil=35           # Upper bound for initial Conv1D filters
min_fil=12           # Lower bound for initial Conv1D filters
max_ker=30           # Maximum width of the convolutional sliding window
min_ker=5            # Minimum width of the convolutional sliding window

echo "====================================="
echo "-- Initializing Random Parameter Sweep"
echo "-- Filter Range: [$min_fil, $max_fil]"
echo "-- Kernel Range: [$min_ker, $max_ker]"
echo "====================================="

# --- EXECUTION ---
# Iteratively samples and trains until manually interrupted (Ctrl+C).

for i in {1..10000};
do
	FILTERS=$(($RANDOM%($max_fil-$min_fil+1)+$min_fil))
	KERNEL_SIZE=$(($RANDOM%($max_ker-$min_ker+1)+$min_ker))

	echo " "
	echo "Run #$i: Training with FILTERS=$FILTERS, KERNEL_SIZE=$KERNEL_SIZE"
	
	python MAIN.py --batch_size $BATCH_SIZE --epochs $EPOCHS --filters $FILTERS \
		--kernel_size $KERNEL_SIZE --initial_lr $INITIAL_LR --min_lr $MIN_LR \
		--patience $PATIENCE --min_delta $MIN_DELTA --verbose $VERBOSE
done

