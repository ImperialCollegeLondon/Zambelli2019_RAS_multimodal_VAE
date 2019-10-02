#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
mkdir replications_FM
cd replications_FM

for i in {1..10}
do
    echo "run $i "
    mkdir run_$i
    cd run_$i
    ln -s ../../matlab ./

    python ../../ff_forward_model.py
    cd ../
done

