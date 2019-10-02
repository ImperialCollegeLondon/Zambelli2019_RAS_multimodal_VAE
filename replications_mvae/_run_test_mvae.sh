#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
for i in {1..10}
do
    echo "run $i "
    cd run_$i
    mkdir results
    python ../test_final_completeloss.py 1
    
    cd ../
done

