#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
mkdir replications_droniou_martina
cd replications_droniou_martina
ln -s ../matlab ./
for i in {9..10}
do
    echo "run $i "
    mkdir run_$i
    cd run_$i


    python ../../comparison_exp/droniou.py 0
    cd ../
done

