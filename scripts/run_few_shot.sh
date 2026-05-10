#!/bin/bash

# cd ..
DATA=/data
TRAINER=CoDA
CFG=b2n
SHOTS=16  # number of shots (1, 2, 4, 8, 16)  # class-specific context (False or True)
FOLDER=few_shot
GPU=0

for DATASET in eurosat dtd fgvc_aircraft food101 oxford_flowers oxford_pets stanford_cars ucf101 caltech101
do
for SEED in 1 2 3
do
    DIR=${FOLDER}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} 
    fi
done
done


CFG=b2n
for DATASET in imagenet sun397
do
for SEED in 1 2 3
do
    DIR=${FOLDER}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        DATASET.NUM_SHOTS ${SHOTS} 
    fi
done
done
