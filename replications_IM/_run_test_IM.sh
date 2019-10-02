#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
for i in {1..10}
do
    echo "run $i "
    cd run_$i

    cp ../../replications_FM/run_$i/models/* ./models/
    
    mkdir results
    
    python ../test_ff_fm_im.py
    
    cd ../
done

