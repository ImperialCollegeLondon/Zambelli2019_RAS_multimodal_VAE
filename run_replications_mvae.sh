#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
mkdir replications_mvae
cd replications_mvae

for i in {9..10}
do
    echo "run $i "
    mkdir run_$i
    cd run_$i
    ln -s ../../matlab ./

    python ../../train_final_completeloss.py 1
    cd ../
done

