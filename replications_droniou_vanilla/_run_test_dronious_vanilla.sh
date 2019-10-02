#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
for i in {1..10}
do
    echo "run $i "
    cd run_$i
    mkdir results
    python ../test_droniou_vanilla.py 1
    python ../test_droniou_vanilla.py 2
    python ../test_droniou_vanilla.py 3
    python ../test_droniou_vanilla.py 4

    
    cd ../
done

