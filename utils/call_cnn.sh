#! /bin/bash

# $1: The number of networks to be run

for ((i = 0; i < $1; i++)); do
	python cnn.py model_2_cnn_1 --dataset_index=$i >> log.txt
done

for ((i = 0; i < $1; i++)); do
	python cnn.py model_2_cnn_2 --dataset_index=$i >> log.txt
done

for ((i = 0; i < $1; i++)); do
	python cnn.py model_2_cnn_3 --dataset_index=$i >> log.txt
done

