#!/bin/bash

echo "=> Generate data.."

cd data || exit

rm -rf mnist/all_data

python generate_data.py \
  --dataset_name mnist \
  --n_clients 10 \
  --non_iid \
  --n_classes_per_client 2 \
  --frac 0.2 \
  --save_dir mnist \
  --seed 1234

cd ..

echo "=> Train.."

# EXERCISE 4
# for i in 1 5 10 50 100
# do
#   echo "==> Training with local_steps=$i"
#   python train.py \
#     --experiment "mnist" \
#     --n_rounds 100 \
#     --local_steps $i \
#     --local_optimizer sgd \
#     --local_lr 0.001 \
#     --server_optimizer sgd \
#     --server_lr 0.1 \
#     --bz 128 \
#     --device "cpu" \
#     --log_freq 1 \
#     --verbose 1 \
#     --logs_dir "logs/mnist/local_steps_${i}/" \
#     --seed 12

echo "=> Train with uniform client sampling (without replacement).."

local_steps=5
# EXERCISE 5.1
# different sampling rate
# for frac in 0.2 0.5 1.0
# do
#   echo "==> Training with sampling_rate=$frac"
#   python train.py \
#     --experiment "mnist" \
#     --n_rounds 100 \
#     --local_steps $local_steps \
#     --local_optimizer sgd \
#     --local_lr 0.001 \
#     --server_optimizer sgd \
#     --server_lr 0.1 \
#     --bz 128 \
#     --device "cpu" \
#     --sampling_rate $frac \
#     --log_freq 1 \
#     --verbose 1 \
#     --logs_dir "logs/ex5_1/sampling_${frac}/" \
#     --seed 12

# # EXERCISE 5.2

# echo "=> Train with sampling with replacement.."

# for frac in 0.2 0.5 1.0
# do
#   echo "==> Training with sampling_rate=$frac (with replacement)"
#   python train.py \
#     --experiment "mnist" \
#     --n_rounds 100 \
#     --local_steps $local_steps \
#     --local_optimizer sgd \
#     --local_lr 0.001 \
#     --server_optimizer sgd \
#     --server_lr 0.1 \
#     --bz 128 \
#     --device "cpu" \
#     --sampling_rate $frac \
#     --sample_with_replacement \
#     --log_freq 1 \
#     --verbose 1 \
#     --logs_dir "logs/ex5_2/sampling_${frac}/" \
#     --seed 12

# EXERCISE 6.
echo "=> Train FedProx (Tackling Data Heterogeneity).."

local_steps_list="1 5 10 50 100"

for i in $local_steps_list
do
  echo "==> Training FedProx with local_steps=$i"
  python train.py \
    --experiment "mnist" \
    --algorithm fedprox \
    --n_rounds 100 \
    --local_steps $i \
    --local_optimizer prox_sgd \
    --mu 2 \
    --local_lr 0.001 \
    --server_optimizer sgd \
    --server_lr 0.1 \
    --bz 128 \
    --device "cpu" \
    --log_freq 1 \
    --verbose 1 \
    --logs_dir "logs/ex6_1/fedprox/local_steps_${i}/" \
    --seed 12

done
# EXERCISE 7.1
# echo "=> Train SCAFFOLD (Mitigating Client Drift).."

# local_steps_list="1 5 10 50"

# for i in $local_steps_list
# do
#   echo "==> Training SCAFFOLD with local_steps=$i"
#   python train.py \
#     --experiment "mnist" \
#     --n_rounds 100 \
#     --local_steps $i \
#     --local_optimizer sgd \
#     --server_optimizer sgd \
#     --server_lr 0.1 \
#     --local_lr 0.001 \
#     --bz 128 \
#     --device "cpu" \
#     --algorithm scaffold \
#     --log_freq 1 \
#     --verbose 1 \
#     --logs_dir "logs/ex7_1/scaffold/local_steps_${i}/" \
#     --seed 12

