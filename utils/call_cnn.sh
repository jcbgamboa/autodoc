#! /bin/bash

# $1: The number of networks to be run

# --------------------------------------- Model 5 with Dropout
for ((i = 0; i < $1; i++)); do
	python cnn.py model_5_cnn_dropout_nomean --dataset_index=$i >> log.txt
done

# --------------------------------------- Model 5 without Dropout
for ((i = 0; i < $1; i++)); do
	python cnn.py model_5_cnn_nomean --dataset_index=$i >> log.txt
done

# --------------------------------------- Pretrain Model 6
python cnn.py model_6 --run_as_caes --dataset=rvl-cdip --no_test >> log.txt

# --------------------------------------- Model 6 without Dropout
for ((i = 0; i < $1; i++)); do
	python cnn.py model_6_cnn --dataset_index=$i >> log.txt
done

# --------------------------------------- Model 6 with Dropout
for ((i = 0; i < $1; i++)); do
	python cnn.py model_6_cnn_dropout --dataset_index=$i >> log.txt
done

# Won't do "Transfer Learning" (in the strict sense) because it is crashing

# --------------------------------------- Model 5 with Dropout
#for ((i = 0; i < $1; i++)); do
#	python cnn.py model_5_cnn_dropout --use_mean_image --dataset_index=$i >> log.txt
#done

# --------------------------------------- Model 5 without Dropout
#for ((i = 0; i < $1; i++)); do
#	python cnn.py model_5_cnn --use_mean_image --dataset_index=$i >> log.txt
#done

# --------------------------------------- Model 2 without Dropout
#for ((i = 0; i < $1; i++)); do
#	python cnn.py model_2_cnn --dataset_index=$i >> log.txt
#done


