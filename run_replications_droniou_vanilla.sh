#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
mkdir replications_droniou_vanilla
cd replications_droniou_vanilla
ln -s ../matlab ./
for i in {1..10}
do
    echo "run $i "
    mkdir run_$i
    cd run_$i


    python ../../comparison_exp/droniou.py 1
    cd ../
done

