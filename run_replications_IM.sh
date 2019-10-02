#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
mkdir replications_IM
cd replications_IM

for i in {1..10}
do
    echo "run $i "
    mkdir run_$i
    cd run_$i
    ln -s ../../matlab ./

    python ../../ff_inverse_model.py
    cd ../
done

