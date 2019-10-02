#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
mkdir replications_vae_martina
cd replications_vae_martina

for i in {1..10}
do
    echo "run $i "
    mkdir run_$i
    cd run_$i
    ln -s ../../matlab ./

    python ../../vanilla_vae.py 0
    cd ../
done

