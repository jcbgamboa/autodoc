#! /bin/bash

for ((i = 0; i < 100; i++)); do
	# Choose a random number
	RANDN=$(( ( RANDOM % 80 )  + 20 ))

	# Generates subset of tobacco dataset
	python utils/generate_random_dataset.py data/datasets/tobacco/train.txt \
		$RANDN data/datasets/tobacco/train_$((i)).txt
done
