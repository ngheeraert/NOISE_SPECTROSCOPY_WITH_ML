# =============================================================================
# Parameter sweep launcher for training runs.
#
# This script repeatedly calls MAIN.py with a fixed set of training hyperparameters
# while randomly sampling two architecture parameters:
#   - FILTERS: number of convolution filters in the first block
#   - KERNEL_SIZE: size of the 1D convolution kernel
#
# Notes:
#   - The loop count is hard-coded (1..10000). Stop with Ctrl+C when desired.
# =============================================================================

#FILTERS=28
#KERNEL_SIZE=30
BATCH_SIZE=64
EPOCHS=20
INITIAL_LR=0.001
MIN_LR=0.00005
PATIENCE=4
MIN_DELTA=0.5
VERBOSE=False

max_fil=35
min_fil=12
max_ker=30
min_ker=5

echo "====================================="
echo "-- Number of parameter sets = $nb"
echo "-- min filters = $min_fil"
echo "-- max filters = $max_fil"
echo "-- min kernel size = $min_ker"
echo "-- max kernel size = $max_ker"
echo "====================================="

for i in {1..10000};
do
	FILTERS=$(($RANDOM%($max_fil-$min_fil+1)+$min_fil))
	KERNEL_SIZE=$(($RANDOM%($max_ker-$min_ker+1)+$min_ker))

	echo " "
	python MAIN.py --batch_size $BATCH_SIZE --epochs $EPOCHS --filters $FILTERS \
		--kernel_size $KERNEL_SIZE --initial_lr $INITIAL_LR --min_lr $MIN_LR \
		--patience $PATIENCE --min_delta $MIN_DELTA --verbose $VERBOSE

done

