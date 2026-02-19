BATCH_SIZE=64
EPOCHS=150
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


