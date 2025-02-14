#!/bin/bash

experiments=('product' 'summation' 'negation')
suffixes=('a1' 'a2' 'b1' 'b2' 'c1' 'c2')

for exp in "${experiments[@]}"
do
    for suffix in "${suffixes[@]}"
    do
        experiment_name="${exp}_${suffix}"
        # if there is error, remove the "> /dev/null 2>&1" and run it in terminal to debug
        python ddpm.py --experiment_name "$experiment_name" --dataset "$experiment_name" --num_epochs 1000 --mlp_type energy > /dev/null 2>&1
    done
done
