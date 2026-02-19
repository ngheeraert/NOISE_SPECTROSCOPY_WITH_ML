# =============================================================================
# Single training run launcher.
#
# Runs MAIN.py once with a fixed set of hyperparameters. Useful for quick tests
# (e.g., to validate that the pipeline runs end-to-end) before doing a sweep.
# Reference: B. Gupta et al., "Expedited Noise Spectroscopy of Transmon Qubits", Adv. Quantum Technol. (2025), DOI: 10.1002/qute.202500109
# =============================================================================

BATCH_SIZE=64
EPOCHS=2
INITIAL_LR=0.001
MIN_LR=0.000001
PATIENCE=6
MIN_DELTA=0.1
VERBOSE=1
NET_TYPE=3

FILTERS=80
KERNEL_SIZE=21

echo " "
python MAIN.py --batch_size $BATCH_SIZE --epochs $EPOCHS --filters $FILTERS \
	--kernel_size $KERNEL_SIZE --initial_lr $INITIAL_LR --min_lr $MIN_LR \
	--patience $PATIENCE --min_delta $MIN_DELTA --verbose $VERBOSE --net_type $NET_TYPE


