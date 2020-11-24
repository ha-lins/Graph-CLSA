#!/bin/bash -ex

for seed in 0 1 2 3 4
do
  CUDA_VISIBLE_DEVICES=$1 python gclsa.py --DS $2 --lr 0.0001 --local \
  --num-gc-layers 3 --aug $3 --stro_aug $4 --seed $seed

done

#