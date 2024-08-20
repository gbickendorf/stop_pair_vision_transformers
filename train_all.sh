#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate torch

mkdir log
for modelname in MaxVit CoAtNet
do
    for jetdef in AK14 AK10 AK08
    do
        python train.py $jetdef $modelname 0 | tee log/${jetdef}_${modelname}.log #2>log2/${jetdef}_${modelname}_${i}.err
        echo ${modelname} ${jetdef} > completed.txt
    done
done